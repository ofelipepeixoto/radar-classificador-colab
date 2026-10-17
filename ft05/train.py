"""Explicit, non-quantized Transformers + PEFT LoRA training on a Colab GPU.

Importing this module stays offline and does not import the GPU stack.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

from ft05 import load, split, evaluate, compare, load_frozen, strict_json
from adapter_integrity import adapter_hashes
from dependencies import BACKEND as TRAINING_BACKEND, VERSIONS

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '5fee7c4ed634dc66c6e318c8ac2897b8b9154536'
PACKAGES = tuple(VERSIONS)
MAX_CONTEXT = 2048
MAX_NEW_TOKENS = 96
MIN_GPU_BYTES = 16_000_000_000  # 16 GB decimal; includes class 16 GB GPUs reporting ~15 GiB.
SEED = 3407
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def check_config(config):
    if config.get('model') != MODEL or config.get('model_revision') != REVISION:
        raise ValueError('pin_model_revision')
    if config.get('synthetic_only') is not True:
        raise ValueError('synthetic_only')
    if config.get('gpu_budget_approved') is not True or config.get('dependency_reviewed') is not True:
        raise ValueError('gpu_and_dependency_review_required')
    if config.get('gguf_export_approved') is True:
        raise ValueError('gguf_export_not_supported: este backend não exporta GGUF')
    versions = config.get('package_versions')
    if type(versions) is not dict or versions != VERSIONS:
        raise ValueError('pin_all_package_versions')
    return versions


def cuda_runtime():
    """Fail before model downloads unless the selected GPU meets the recipe."""
    import torch
    if not torch.cuda.is_available():
        raise ValueError('cuda_required: selecione uma GPU no Colab')
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < MIN_GPU_BYTES:
        raise ValueError('gpu_memory_too_small: esta receita requer uma GPU de classe 16 GB ou maior')
    # Allocation/emulation alone is insufficient for the fused training kernels.
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    return {'device': 'cuda:0', 'name': properties.name,
            'total_memory_bytes': properties.total_memory,
            'compute_capability': list(torch.cuda.get_device_capability(0)),
            'precision': 'bf16' if bf16 else 'fp16',
            'dtype': torch.bfloat16 if bf16 else torch.float16}


def load_pinned_model(revision, *, runtime=None):
    """Load the exact base and tokenizer, without remote code or quantization."""
    if revision != REVISION:
        raise ValueError('pin_model_revision')
    runtime = cuda_runtime() if runtime is None else runtime
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=revision, trust_remote_code=False, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, trust_remote_code=False, use_safetensors=True,
        dtype=runtime['dtype'], attn_implementation='sdpa', device_map=None)
    loaded_name = getattr(model.config, '_name_or_path', None)
    if not isinstance(loaded_name, str) or loaded_name.lower() != MODEL.lower():
        raise ValueError(f'model_source_changed: {loaded_name!r}')
    if (getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False)
            or getattr(model.config, 'quantization_config', None) is not None):
        raise ValueError('quantized_base_not_supported')
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError('chat_template_missing')
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('padding_and_eos_tokens_missing')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = MAX_CONTEXT
    model.to(runtime['device'])
    model.eval()
    return model, tokenizer, loaded_name


def generation_config(model):
    """Preserve base generation settings and use one explicit new-token cap.

    With max_length=None, GenerationMixin derives total length from the input
    and max_new_tokens; no inherited total-length cap competes with it.
    """
    from transformers import GenerationConfig
    config = GenerationConfig.from_dict(model.generation_config.to_dict())
    config.max_length = None
    config.max_new_tokens = MAX_NEW_TOKENS
    config.do_sample = False
    config.use_cache = True
    return config


def generate_response(model, tokenizer, messages):
    """Generate text and synchronized latency; reject long input intact."""
    import torch
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False, truncation=False)
    input_length = inputs['input_ids'].shape[-1]
    if input_length + MAX_NEW_TOKENS > MAX_CONTEXT:
        raise ValueError('message_context_too_long: reduza a mensagem; nenhum texto foi truncado')
    inputs = inputs.to(model.device)
    on_cuda = str(model.device).startswith('cuda')
    if on_cuda:
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, generation_config=generation_config(model))
    if on_cuda:
        torch.cuda.synchronize(model.device)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    text = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True)
    return text, elapsed_ms


def infer(model, tokenizer, rows):
    model.eval()
    predictions = {}
    for row in rows:
        # Gold assistant answers are never passed into generation.
        predictions[row['id']], _ = generate_response(model, tokenizer, row['messages'][:-1])
    score = evaluate(rows, predictions)
    score['model_inference_executed'] = True
    return score, predictions


def build_sft_config(output, runtime):
    """Construct the exact pinned TRL recipe, separately testable without CUDA."""
    from trl import SFTConfig
    return SFTConfig(output_dir=str(Path(output) / 'checkpoints'), dataset_text_field='text',
        max_length=MAX_CONTEXT, per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=3, learning_rate=2e-4, optim='adamw_torch',
        bf16=runtime['precision'] == 'bf16', fp16=runtime['precision'] == 'fp16',
        use_cpu=runtime['device'] == 'cpu',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to='none', seed=SEED, data_seed=SEED, packing=False,
        save_strategy='epoch', save_total_limit=2)


def checked_training_metrics(train_output, trainer):
    """Reject an incomplete/nonfinite run before declaring a saved model trained."""
    step = getattr(train_output, 'global_step', None)
    loss = getattr(train_output, 'training_loss', None)
    if type(step) is not int or step <= 0:
        raise ValueError('training_did_not_complete_a_step')
    if type(loss) not in (int, float) or not math.isfinite(loss):
        raise ValueError('training_loss_not_finite')
    metrics = getattr(train_output, 'metrics', None)
    if type(metrics) is not dict:
        raise ValueError('training_metrics_missing')
    history = getattr(getattr(trainer, 'state', None), 'log_history', [])
    observed = []
    for row in [metrics, *history]:
        values = {}
        for key in ('loss', 'train_loss', 'grad_norm'):
            if key in row and row[key] is not None:
                value = row[key]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('nonfinite_training_metric_' + key)
                values[key] = value
        if values:
            observed.append(values)
    return {'global_step': step, 'training_loss': loss, 'metrics': metrics,
            'observed_loss_and_gradient_logs': observed,
            'gradient_logs_checked': sum('grad_norm' in row for row in observed)}


def run(dataset, config_path, output, execute=False, frozen_dir=None):
    rows = load(dataset)
    parts = split(rows)
    plan = {'mode': 'plan', 'counts': {k: len(v) for k, v in parts.items()},
            'model': MODEL, 'training_backend': TRAINING_BACKEND,
            'network_executed': False, 'trained': False}
    if not execute:
        return plan
    if frozen_dir is None:
        raise ValueError('frozen_split_required')
    parts, frozen_sha256 = load_frozen(dataset, frozen_dir)
    config = strict_json(Path(config_path).read_text())
    versions = check_config(config)
    for package, version in versions.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError('package_missing_' + package + ': instale as dependências desta sessão') from exc
        if actual != version:
            raise ValueError('package_version_mismatch_' + package)
    output = Path(output)
    if output.exists():
        raise ValueError('output_already_exists')
    runtime = cuda_runtime()
    from transformers import set_seed
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    set_seed(SEED)
    model, tokenizer, loaded_name = load_pinned_model(config['model_revision'], runtime=runtime)
    texts = {name: [tokenizer.apply_chat_template(row['messages'], tokenize=False,
                      add_generation_prompt=False) for row in records]
             for name, records in parts.items()}
    if any(len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']) > MAX_CONTEXT
           for group in texts.values() for text in group):
        raise ValueError('sequence_would_be_truncated')
    output.mkdir(parents=True)
    baseline, before = infer(model, tokenizer, parts['test'])
    (output / 'model-baseline.json').write_text(json.dumps(baseline, indent=2))
    (output / 'predictions-model-original.json').write_text(json.dumps(before, indent=2))
    lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias='none',
                      target_modules=LORA_TARGETS, task_type='CAUSAL_LM',
                      base_model_name_or_path=MODEL, revision=config['model_revision'])
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    args = build_sft_config(output, runtime)
    trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_dict({'text': texts['train']}),
        eval_dataset=Dataset.from_dict({'text': texts['validation']}))
    train_output = trainer.train()
    training_metrics = checked_training_metrics(train_output, trainer)
    training_metrics_file = output / 'training-metrics.json'
    training_metrics_file.write_text(json.dumps(training_metrics, indent=2, allow_nan=False))
    validation = trainer.evaluate()
    validation_loss = validation.get('eval_loss')
    if type(validation_loss) not in (int, float) or not math.isfinite(validation_loss):
        raise ValueError('validation_loss_not_finite')
    (output / 'validation.json').write_text(json.dumps(validation, indent=2, allow_nan=False))
    finetuned, after = infer(model, tokenizer, parts['test'])
    (output / 'model-finetuned.json').write_text(json.dumps(finetuned, indent=2))
    (output / 'predictions-model-personalized.json').write_text(json.dumps(after, indent=2))
    comparison = compare(parts['test'], before, after)
    comparison.update(original=baseline, personalized=finetuned, model_inference_executed=True)
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    # Vocabulary is unchanged; avoid PEFT's optional remote base-config lookup.
    model.save_pretrained(str(output / 'adapter'), safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(str(output / 'adapter'))
    adapter_artifacts = adapter_hashes(output / 'adapter', MODEL, config['model_revision'])
    (output / 'chat-template.txt').write_text(tokenizer.chat_template)
    frozen_manifest = strict_json((Path(frozen_dir) / 'manifest.json').read_text())
    manifest = {'model': MODEL, 'revision': config['model_revision'],
        'loaded_model_name': loaded_name, 'exact_model_name_required': True,
        'training_backend': TRAINING_BACKEND, 'precision': runtime['precision'],
        'quantization': False, 'compute': {k: v for k, v in runtime.items() if k != 'dtype'},
        'seed': SEED, 'max_context': MAX_CONTEXT, 'max_new_tokens': MAX_NEW_TOKENS,
        'training_global_step': training_metrics['global_step'],
        'training_loss': training_metrics['training_loss'],
        'training_metrics_sha256': hashlib.sha256(training_metrics_file.read_bytes()).hexdigest(),
        'versions': versions, 'adapter_artifacts_sha256': adapter_artifacts,
        'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'frozen_manifest_sha256': frozen_sha256,
        'review_mode': frozen_manifest['review_mode'], 'human_reviewed': frozen_manifest['human_reviewed'],
        'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        'test_ids': [row['id'] for row in parts['test']],
        'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'trained': True, 'gguf_export_requested': False,
        'ollama_tested': False, 'llama_cpp_tested': False}
    (output / 'run.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--config', default='training-config.json')
    parser.add_argument('--output', default='training-run')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--frozen', help='Directory created by the corpus preparation step')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.config, args.output, args.train, frozen_dir=args.frozen), indent=2))
