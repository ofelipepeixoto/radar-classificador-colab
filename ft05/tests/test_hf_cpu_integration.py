"""Optional real HF/PEFT/TRL integration; tiny random model, CPU, no downloads.

RUN_HF_CPU_INTEGRATION=1 enables this dependency-heavy check. It does not
validate Colab CUDA, the pinned 1.5B model, or classification quality.
"""
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('RUN_HF_CPU_INTEGRATION') == '1',
                     'Optional real HF CPU integration is disabled')
class HfCpuIntegrationTests(unittest.TestCase):
    @patch('socket.socket.connect', side_effect=AssertionError('CPU integration must stay offline'))
    def test_tiny_lora_backprop_saves_reloads_and_generates_without_length_conflict(self, connect):
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, set_seed
        from trl import SFTTrainer
        from adapter_integrity import adapter_hashes, verify_adapter
        from train import (MODEL, REVISION, LORA_TARGETS, generate_response,
                           build_sft_config, MAX_NEW_TOKENS, MAX_CONTEXT)

        set_seed(3407)
        vocab = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2, 'pedido': 3,
                 'entrega': 4, 'media': 5, 'user': 6, 'assistant': 7}
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' [EOS] ' }}"
            "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
        config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048, eos_token_id=1, pad_token_id=0,
            attention_dropout=0.0)
        base = Qwen2ForCausalLM(config)
        base.config._name_or_path = MODEL
        base.name_or_path = MODEL
        initial = {name: value.detach().clone() for name, value in base.state_dict().items()}
        lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias='none',
            target_modules=LORA_TARGETS, task_type='CAUSAL_LM',
            base_model_name_or_path=MODEL, revision=REVISION)
        model = get_peft_model(base, lora)
        before = {name: value.detach().clone() for name, value in model.named_parameters()
                  if value.requires_grad}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # TRL 1.15's fused head needs Triton/CUDA; exercise the actual HF +
            # PEFT save/reload and gradient boundary on CPU without mocking it.
            model.train()
            batch = tokenizer('user pedido entrega [EOS] assistant entrega media [EOS]',
                              return_tensors='pt')
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
            loss = model(**batch, labels=batch['input_ids']).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.assertTrue(any(not torch.equal(before[name], value.detach())
                                for name, value in model.named_parameters() if name in before))
            # Validate real TRL + Datasets constructor/preparation APIs. Its
            # fused training kernel is CUDA-only and is not exercised here.
            args = build_sft_config(root, {'device': 'cpu', 'precision': 'fp32'})
            self.assertEqual(args.max_length, MAX_CONTEXT)
            self.assertEqual(args.optim.value, 'adamw_torch')
            self.assertFalse(args.fp16)
            self.assertFalse(args.bf16)
            self.assertTrue(args.gradient_checkpointing)
            trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                train_dataset=Dataset.from_dict({'text': [
                    'user pedido entrega [EOS] assistant entrega media [EOS]']}))
            self.assertEqual(len(trainer.train_dataset), 1)
            self.assertIn('input_ids', trainer.train_dataset.column_names)
            model.eval()
            adapter = root / 'adapter'
            model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
            hashes = adapter_hashes(adapter, MODEL, REVISION)
            self.assertEqual(verify_adapter(adapter, hashes, MODEL, REVISION), hashes)
            restored_base = Qwen2ForCausalLM(config)
            restored_base.load_state_dict(initial)
            restored_base.config._name_or_path = MODEL
            restored_base.name_or_path = MODEL
            restored = PeftModel.from_pretrained(restored_base, adapter, is_trainable=False)
            restored.eval()
            inputs = tokenizer('pedido entrega', return_tensors='pt')
            with torch.inference_mode():
                torch.testing.assert_close(model(**inputs).logits, restored(**inputs).logits)

            records = []

            class Capture(logging.Handler):
                def emit(self, record):
                    records.append(record.getMessage())

            logger = logging.getLogger('transformers.generation.utils')
            handler = Capture()
            logger.addHandler(handler)
            try:
                text, elapsed = generate_response(restored, tokenizer,
                    [{'role': 'user', 'content': 'pedido entrega'}])
            finally:
                logger.removeHandler(handler)
            self.assertIsInstance(text, str)
            self.assertGreaterEqual(elapsed, 0)
            self.assertFalse(any('max_new_tokens' in message and 'max_length' in message
                                 for message in records), records)
            self.assertLessEqual(len(tokenizer(text)['input_ids']), MAX_NEW_TOKENS)
        connect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
