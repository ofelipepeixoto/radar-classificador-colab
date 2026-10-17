"""Exercise training and UI wiring with CPU doubles; no model downloads."""
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auto_pilot import write_auto_dataset
from ft05 import evaluate, prepare
from personal_smoke import REVISION, preflight
from train import MODEL, TRAINING_BACKEND, MIN_GPU_BYTES, run
from dependencies import VERSIONS


class RuntimeGuardsTests(unittest.TestCase):
    def test_training_manifest_seals_the_actual_saved_adapter(self):
        lora_calls, trainer_calls = [], []

        class Model:
            config = SimpleNamespace(use_cache=True)

            def save_pretrained(self, directory, safe_serialization, save_embedding_layers):
                self.safe_serialization = safe_serialization
                self.save_embedding_layers = save_embedding_layers
                directory = Path(directory)
                directory.mkdir()
                (directory / 'adapter_config.json').write_text(json.dumps({
                    'base_model_name_or_path': MODEL, 'peft_type': 'LORA', 'revision': REVISION}))
                (directory / 'adapter_model.safetensors').write_bytes(b'CPU-double-weights')

        class Tokenizer:
            chat_template = 'CPU-double-template'

            def apply_chat_template(self, *args, **kwargs):
                return 'short synthetic text'

            def __call__(self, *args, **kwargs):
                return {'input_ids': [1]}

            def save_pretrained(self, directory):
                pass

        class Trainer:
            def __init__(self, **kwargs):
                trainer_calls.append(kwargs)
                self.state = SimpleNamespace(log_history=[{'loss': 0.1, 'grad_norm': 0.2}])

            def train(self):
                return SimpleNamespace(global_step=3, training_loss=0.1, metrics={'train_loss': 0.1})

            def evaluate(self):
                return {'eval_loss': 0.0}

        def infer_stub(model, tokenizer, rows):
            predictions = {row['id']: row['messages'][-1]['content'] for row in rows}
            return evaluate(rows, predictions), predictions

        def lora_config(**kwargs):
            lora_calls.append(kwargs)
            return kwargs

        model, tokenizer = Model(), Tokenizer()
        modules = {
            'transformers': SimpleNamespace(set_seed=lambda seed: None),
            'peft': SimpleNamespace(LoraConfig=lora_config, get_peft_model=lambda base, config: base),
            'datasets': SimpleNamespace(Dataset=SimpleNamespace(from_dict=lambda data: data)),
            'trl': SimpleNamespace(SFTTrainer=Trainer, SFTConfig=lambda **kwargs: kwargs),
        }
        runtime = {'device': 'cuda:0', 'name': 'CPU-double', 'precision': 'bf16',
                   'dtype': 'bf16', 'total_memory_bytes': MIN_GPU_BYTES,
                   'compute_capability': [8, 0]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, frozen, output = root / 'corpus', root / 'frozen', root / 'run'
            write_auto_dataset(corpus)
            prepare(corpus, frozen, automated=True)
            config = root / 'config.json'
            config.write_text(json.dumps({'model': MODEL, 'model_revision': REVISION,
                'synthetic_only': True, 'gpu_budget_approved': True,
                'dependency_reviewed': True, 'package_versions': dict(VERSIONS)}))
            with patch.dict('sys.modules', modules), \
                    patch('train.importlib.metadata.version', side_effect=VERSIONS.__getitem__), \
                    patch('train.cuda_runtime', return_value=runtime), \
                    patch('train.load_pinned_model', return_value=(model, tokenizer, MODEL)), \
                    patch('train.infer', side_effect=infer_stub):
                manifest = run(corpus, config, output, execute=True, frozen_dir=frozen)
            recorded = json.loads((output / 'run.json').read_text())
            self.assertEqual(recorded, manifest)
            self.assertEqual(manifest['training_backend'], TRAINING_BACKEND)
            self.assertFalse(manifest['quantization'])
            self.assertEqual(manifest['precision'], 'bf16')
            self.assertEqual(manifest['training_global_step'], 3)
            self.assertTrue((output / 'training-metrics.json').is_file())
            self.assertIs(model.safe_serialization, True)
            self.assertIs(model.save_embedding_layers, False)
            self.assertEqual(lora_calls[0]['lora_dropout'], 0.0)
            self.assertEqual(lora_calls[0]['task_type'], 'CAUSAL_LM')
            self.assertEqual(trainer_calls[0]['args']['optim'], 'adamw_torch')
            self.assertIs(trainer_calls[0]['args']['bf16'], True)
            self.assertIs(trainer_calls[0]['args']['fp16'], False)
            for name, expected in manifest['adapter_artifacts_sha256'].items():
                self.assertEqual(expected, hashlib.sha256((output / 'adapter' / name).read_bytes()).hexdigest())
            self.assertEqual(preflight(output, corpus, frozen, versions=False)['meta'], manifest)

    def test_ui_rejects_sensitive_text_and_keeps_button_usable(self):
        from personal_ui import launch

        class Output:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def clear_output(self):
                pass

        class Button:
            disabled = False

            def __init__(self, **kwargs):
                pass

            def on_click(self, callback):
                self.callback = callback

        widgets = SimpleNamespace(Textarea=lambda **kwargs: SimpleNamespace(value=''),
            Button=Button, Output=Output, Layout=lambda **kwargs: kwargs, VBox=lambda items: items)
        modules = {'ipywidgets': widgets, 'IPython.display': SimpleNamespace(display=lambda obj: None)}
        with patch.dict('sys.modules', modules), \
                patch('personal_smoke.preflight', return_value={}), \
                patch('personal_smoke.load_saved_adapter', return_value=(object(), object(), MODEL)), \
                patch('train.generate_response', return_value=(
                    '{"categoria":"entrega","urgencia":"media"}', 5.0)) as generate, \
                redirect_stdout(io.StringIO()) as messages:
            box, button, _ = launch('/unused', '/unused', '/unused')
            box.value = 'Escreva para pessoa@example.com'
            button.callback(None)
            self.assertIn('possible_sensitive_data', messages.getvalue())
            self.assertFalse(button.disabled)
            generate.assert_not_called()
            box.value = 'Minha remessa está com a etiqueta borrada, sem prazo definido.'
            button.callback(None)
            self.assertIn('Categoria: entrega', messages.getvalue())
            self.assertFalse(button.disabled)
            generate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
