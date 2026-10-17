import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from dependencies import VERSIONS
from train import (MODEL, REVISION, MIN_GPU_BYTES, MAX_CONTEXT, MAX_NEW_TOKENS,
                   check_config, cuda_runtime, generate_response, infer)


class HfBackendTests(unittest.TestCase):
    def test_gpu_gate_precedes_model_downloads_and_selects_precision(self):
        for bf16 in (True, False):
            fake = SimpleNamespace(bfloat16='bf16', float16='fp16', cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda device: SimpleNamespace(
                    name='GPU-double', total_memory=MIN_GPU_BYTES),
                is_bf16_supported=lambda: bf16, get_device_capability=lambda device: (8, 0)))
            with self.subTest(bf16=bf16), patch.dict('sys.modules', {'torch': fake}):
                self.assertEqual(cuda_runtime()['precision'], 'bf16' if bf16 else 'fp16')
        fake.cuda.is_available = lambda: False
        with patch.dict('sys.modules', {'torch': fake}):
            with self.assertRaisesRegex(ValueError, 'cuda_required'):
                cuda_runtime()
        fake.cuda.is_available = lambda: True
        fake.cuda.get_device_properties = lambda device: SimpleNamespace(
            name='small', total_memory=MIN_GPU_BYTES - 1)
        with patch.dict('sys.modules', {'torch': fake}):
            with self.assertRaisesRegex(ValueError, 'gpu_memory_too_small'):
                cuda_runtime()

    def test_export_and_unsafe_dependency_pins_are_refused(self):
        config = {'model': MODEL, 'model_revision': REVISION, 'synthetic_only': True,
                  'gpu_budget_approved': True, 'dependency_reviewed': True,
                  'package_versions': dict(VERSIONS)}
        self.assertEqual(check_config(config), VERSIONS)
        with self.assertRaisesRegex(ValueError, 'gguf_export_not_supported'):
            check_config({**config, 'gguf_export_approved': True})
        with self.assertRaisesRegex(ValueError, 'pin_all_package_versions'):
            check_config({**config, 'package_versions': {**VERSIONS, 'datasets': '4.8.5'}})

    def test_generation_refuses_overflow_instead_of_truncating(self):
        calls, sync = [], []

        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            length = MAX_CONTEXT - MAX_NEW_TOKENS

            def apply_chat_template(self, messages, **kwargs):
                return 'prompt'

            def __call__(self, prompt, **kwargs):
                self.last_options = kwargs
                return Inputs(input_ids=SimpleNamespace(shape=(1, self.length)))

            def decode(self, output, **kwargs):
                return 'generated'

        tokenizer = Tokenizer()
        model = SimpleNamespace(device='cuda:0', generate=lambda **kwargs: calls.append(kwargs)
                                or [[0] * (tokenizer.length + 1)])
        fake_torch = SimpleNamespace(inference_mode=nullcontext,
            cuda=SimpleNamespace(synchronize=lambda device: sync.append(device)))
        with patch.dict('sys.modules', {'torch': fake_torch}), \
                patch('train.generation_config', return_value='explicit-generation-config'):
            result, _ = generate_response(model, tokenizer, [{'role': 'user', 'content': 'fictício'}])
            self.assertEqual(result, 'generated')
            self.assertEqual(sync, ['cuda:0', 'cuda:0'])
            self.assertIs(tokenizer.last_options['truncation'], False)
            self.assertEqual(calls[0]['generation_config'], 'explicit-generation-config')
            tokenizer.length += 1
            with self.assertRaisesRegex(ValueError, 'message_context_too_long'):
                generate_response(model, tokenizer, [{'role': 'user', 'content': 'fictício'}])
            self.assertEqual(len(calls), 1)

    def test_gold_answers_are_not_passed_to_generation(self):
        row = {'id': 'one', 'group_id': 'one', 'synthetic': True, 'reviewed': False,
               'messages': [{'role': 'user', 'content': 'Pedido fictício'},
                            {'role': 'assistant', 'content': '{"categoria":"outro","urgencia":"media"}'}]}
        model = SimpleNamespace(eval=lambda: None)
        with patch('train.generate_response', return_value=(row['messages'][-1]['content'], 1.0)) as generate:
            score, _ = infer(model, object(), [row])
            self.assertEqual(score['exact_match'], 1)
            self.assertEqual(generate.call_args.args[2], row['messages'][:-1])


if __name__ == '__main__':
    unittest.main()
