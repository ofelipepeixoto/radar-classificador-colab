import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapter_integrity import adapter_hashes, verify_adapter
from auto_pilot import write_auto_dataset
from ft05 import prepare
from personal_smoke import REVISION, preflight, sha256
from train import MODEL, PACKAGES


def fixture(root):
    corpus, frozen, run_dir = root / 'corpus.jsonl', root / 'frozen', root / 'run'
    write_auto_dataset(corpus)
    prepare(corpus, frozen, automated=True)
    adapter = run_dir / 'adapter'
    adapter.mkdir(parents=True)
    (adapter / 'adapter_config.json').write_text(json.dumps({
        'base_model_name_or_path': MODEL, 'peft_type': 'LORA', 'revision': None}))
    (adapter / 'adapter_model.safetensors').write_bytes(b'test-fixture-not-a-real-model')
    meta = {'model': MODEL, 'loaded_model_name': MODEL, 'revision': REVISION,
            'exact_model_name_required': True, 'review_mode': 'synthetic_machine_checked',
            'human_reviewed': False, 'trained': True,
            'dataset_sha256': sha256(corpus),
            'frozen_manifest_sha256': sha256(frozen / 'manifest.json'),
            'versions': {name: 'test' for name in PACKAGES},
            'adapter_artifacts_sha256': adapter_hashes(adapter, MODEL, REVISION)}
    (run_dir / 'run.json').write_text(json.dumps(meta))
    return corpus, frozen, run_dir, meta


class AdapterIntegrityTests(unittest.TestCase):
    def test_changed_config_or_weights_rejected_before_gpu_import(self):
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                corpus, frozen, run_dir, _ = fixture(Path(tmp))
                preflight(run_dir, corpus, frozen, versions=False)
                artifact = run_dir / 'adapter' / name
                # Whitespace preserves a valid JSON config but changes its bytes.
                artifact.write_bytes(artifact.read_bytes() + b' ')
                with self.assertRaisesRegex(ValueError, 'adapter_changed_after_training'):
                    preflight(run_dir, corpus, frozen, versions=False)

    def test_legacy_run_is_rejected_with_recovery_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus, frozen, run_dir, meta = fixture(Path(tmp))
            del meta['adapter_artifacts_sha256']
            (run_dir / 'run.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'legado.*notebook original'):
                preflight(run_dir, corpus, frozen, versions=False)

    def test_manifest_cannot_supply_a_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, run_dir, meta = fixture(Path(tmp))
            bad = {**meta['adapter_artifacts_sha256'], '../secret': 'a' * 64}
            with self.assertRaisesRegex(ValueError, 'adapter_integrity_manifest_invalid'):
                verify_adapter(run_dir / 'adapter', bad, MODEL, REVISION)

    def test_mismatched_base_is_rejected_even_with_recomputed_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, run_dir, meta = fixture(Path(tmp))
            config = run_dir / 'adapter' / 'adapter_config.json'
            config.write_text(json.dumps({'base_model_name_or_path': '/tmp/other-model',
                                          'peft_type': 'LORA'}))
            with self.assertRaisesRegex(ValueError, 'adapter_base_mismatch'):
                adapter_hashes(run_dir / 'adapter', MODEL, REVISION)

    def test_missing_dependency_has_readable_instruction(self):
        import importlib.metadata
        with tempfile.TemporaryDirectory() as tmp:
            corpus, frozen, run_dir, _ = fixture(Path(tmp))
            with patch('personal_smoke.importlib.metadata.version',
                       side_effect=importlib.metadata.PackageNotFoundError('unsloth')):
                with self.assertRaisesRegex(ValueError, 'package_missing_unsloth: instale'):
                    preflight(run_dir, corpus, frozen)

    def test_secondary_inference_paths_share_preflight(self):
        from evaluate_saved_adapter import run as stress_run
        from independent_eval import predict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('evaluate_saved_adapter.preflight', side_effect=ValueError('sealed_gate')):
                with self.assertRaisesRegex(ValueError, 'sealed_gate'):
                    stress_run(root / 'run', root / 'out', root / 'corpus', root / 'frozen')
            with patch('independent_eval.sealed', return_value=({}, [], [])), \
                    patch('personal_smoke.preflight', side_effect=ValueError('sealed_gate')):
                with self.assertRaisesRegex(ValueError, 'sealed_gate'):
                    predict(root / 'bundle', root / 'sealed', root / 'run', root / 'out',
                            root / 'corpus', root / 'frozen')


if __name__ == '__main__':
    unittest.main()
