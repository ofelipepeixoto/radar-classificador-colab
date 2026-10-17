"""Exercise runtime wiring with small CPU doubles, without downloading models."""
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auto_pilot import write_auto_dataset
from ft05 import evaluate, prepare
from personal_smoke import REVISION, preflight
from train import MODEL, PACKAGES, run


class RuntimeGuardsTests(unittest.TestCase):
    def test_training_manifest_seals_the_actual_saved_adapter(self):
        class Model:
            def save_pretrained(self, directory):
                directory = Path(directory)
                directory.mkdir()
                (directory / 'adapter_config.json').write_text(json.dumps({
                    'base_model_name_or_path': MODEL, 'peft_type': 'LORA', 'revision': None}))
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
                pass

            def train(self):
                pass

            def evaluate(self):
                return {'eval_loss': 0.0}

        def infer_stub(model, tokenizer, rows):
            predictions = {row['id']: row['messages'][-1]['content'] for row in rows}
            return evaluate(rows, predictions), predictions

        model, tokenizer = Model(), Tokenizer()
        modules = {
            'unsloth': SimpleNamespace(FastLanguageModel=SimpleNamespace(
                get_peft_model=lambda *args, **kwargs: model,
                for_training=lambda model: None)),
            'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
            'datasets': SimpleNamespace(Dataset=SimpleNamespace(from_dict=lambda data: data)),
            'trl': SimpleNamespace(SFTTrainer=Trainer, SFTConfig=lambda **kwargs: kwargs),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, frozen, output = root / 'corpus', root / 'frozen', root / 'run'
            write_auto_dataset(corpus)
            prepare(corpus, frozen, automated=True)
            config = root / 'config.json'
            config.write_text(json.dumps({'model': MODEL, 'model_revision': REVISION,
                'synthetic_only': True, 'gpu_budget_approved': True,
                'dependency_reviewed': True, 'package_versions': {name: 'test' for name in PACKAGES}}))
            with patch.dict('sys.modules', modules), \
                    patch('train.importlib.metadata.version', return_value='test'), \
                    patch('train.load_pinned_model', return_value=(model, tokenizer, MODEL)), \
                    patch('train.infer', side_effect=infer_stub):
                manifest = run(corpus, config, output, execute=True, frozen_dir=frozen)
            recorded = json.loads((output / 'run.json').read_text())
            self.assertEqual(recorded, manifest)
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

        class Inputs(dict):
            def to(self, device):
                return self

        sync_calls = []
        template = 'CPU-widget-template'
        tokenizer = SimpleNamespace(chat_template=template,
            apply_chat_template=lambda *args, **kwargs: 'prompt',
            decode=lambda *args, **kwargs: '{"categoria":"entrega","urgencia":"media"}')

        class Tokenizer:
            chat_template = template
            apply_chat_template = staticmethod(tokenizer.apply_chat_template)
            decode = staticmethod(tokenizer.decode)

            def __call__(self, *args, **kwargs):
                return Inputs(input_ids=SimpleNamespace(shape=[1, 1]))

        widgets = SimpleNamespace(Textarea=lambda **kwargs: SimpleNamespace(value=''),
            Button=Button, Output=Output, Layout=lambda **kwargs: kwargs, VBox=lambda items: items)
        model = SimpleNamespace(eval=lambda: None, device='cuda', generate=lambda **kwargs: [[1, 2]])
        modules = {
            'unsloth': SimpleNamespace(FastLanguageModel=SimpleNamespace(for_inference=lambda model: None)),
            'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                synchronize=lambda: sync_calls.append('sync')), inference_mode=nullcontext),
            'peft': SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)),
            'ipywidgets': widgets, 'IPython.display': SimpleNamespace(display=lambda obj: None),
        }
        checked = {'adapter': Path('/unused'), 'meta': {
            'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest()}}
        with patch.dict('sys.modules', modules), \
                patch('personal_smoke.preflight', return_value=checked), \
                patch('train.load_pinned_model', return_value=(model, Tokenizer(), MODEL)), \
                redirect_stdout(io.StringIO()) as messages:
            box, button, _ = launch('/unused', '/unused', '/unused')
            box.value = 'Escreva para pessoa@example.com'
            button.callback(None)
            self.assertIn('possible_sensitive_data', messages.getvalue())
            self.assertFalse(button.disabled)
            self.assertEqual(sync_calls, [])
            box.value = 'Minha remessa está com a etiqueta borrada, sem prazo definido.'
            button.callback(None)
            self.assertIn('Categoria: entrega', messages.getvalue())
            self.assertFalse(button.disabled)
            self.assertEqual(sync_calls, ['sync', 'sync'])


if __name__ == '__main__':
    unittest.main()
