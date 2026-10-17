import ast
import json
import tempfile
import unittest
from pathlib import Path

from auto_pilot import write_auto_dataset
from ft05 import prepare
from personal_smoke import REVISION, preflight, sha256, validate_messages
from train import MODEL, TRAINING_BACKEND, MIN_GPU_BYTES, MAX_CONTEXT, MAX_NEW_TOKENS
from dependencies import VERSIONS
from adapter_integrity import adapter_hashes


class PersonalSmokeTests(unittest.TestCase):
    def test_new_messages_are_unlabelled_unique_and_screened(self):
        rows = [{'id': 'one', 'text': 'Pedido fictício sobre remessa sem data.'},
                {'id': 'two', 'text': 'Quero consultar uma opção de pagamento.'}]
        self.assertEqual(validate_messages(rows), rows)
        with self.assertRaisesRegex(ValueError, 'expected_id_and_text_only'):
            validate_messages([{**rows[0], 'urgencia': 'alta'}])
        with self.assertRaisesRegex(ValueError, 'duplicate_message_text'):
            validate_messages([rows[0], {'id': 'three', 'text': rows[0]['text']}])
        with self.assertRaisesRegex(ValueError, 'possible_sensitive_data'):
            validate_messages([{'id': 'three', 'text': 'Escreva para pessoa@example.com'}])

    def test_preflight_refuses_changed_source_or_incomplete_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, frozen, run_dir = root / 'corpus.jsonl', root / 'frozen', root / 'run'
            write_auto_dataset(corpus)
            frozen_meta = prepare(corpus, frozen, automated=True)
            (run_dir / 'adapter').mkdir(parents=True)
            (run_dir / 'adapter' / 'adapter_config.json').write_text(json.dumps({
                'base_model_name_or_path': MODEL, 'peft_type': 'LORA', 'revision': None}))
            weights = run_dir / 'adapter' / 'adapter_model.safetensors'
            weights.write_bytes(b'fixture-only')
            meta = {'model': MODEL, 'loaded_model_name': MODEL,
                    'revision': REVISION, 'exact_model_name_required': True,
                    'review_mode': 'synthetic_machine_checked',
                    'human_reviewed': False, 'trained': True,
                    'training_backend': TRAINING_BACKEND, 'precision': 'bf16', 'quantization': False,
                    'compute': {'precision': 'bf16', 'total_memory_bytes': MIN_GPU_BYTES},
                    'max_context': MAX_CONTEXT, 'max_new_tokens': MAX_NEW_TOKENS,
                    'dataset_sha256': sha256(corpus),
                    'frozen_manifest_sha256': sha256(frozen / 'manifest.json'),
                    'versions': dict(VERSIONS)}
            meta['adapter_artifacts_sha256'] = adapter_hashes(run_dir / 'adapter', MODEL, REVISION)
            # load_frozen returns the hash of the exact frozen manifest bytes.
            self.assertEqual(meta['frozen_manifest_sha256'], sha256(frozen / 'manifest.json'))
            (run_dir / 'run.json').write_text(json.dumps(meta))
            self.assertEqual(preflight(run_dir, corpus, frozen, versions=False)['meta'], meta)
            weights.unlink()
            with self.assertRaisesRegex(ValueError, 'adapter_incomplete'):
                preflight(run_dir, corpus, frozen, versions=False)
            weights.write_bytes(b'fixture-only')
            corpus.write_text(corpus.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'training_corpus_changed'):
                preflight(run_dir, corpus, frozen, versions=False)


if __name__ == '__main__':
    unittest.main()
