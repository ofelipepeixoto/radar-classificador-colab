import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from community import paths, prepare_project

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('community_notebook_builder', ROOT / 'scripts/build_notebooks.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class CommunityColabTests(unittest.TestCase):
    def test_notebooks_reproduce_and_default_cells_have_no_external_effects(self):
        digest = hashlib.sha256((ROOT / 'SOURCE-MANIFEST.json').read_bytes()).hexdigest()
        for personal, name in ((False, 'COMECE-AQUI.ipynb'), (True, 'TESTAR-ADAPTADOR.ipynb')):
            nb = json.loads((ROOT / name).read_text())
            self.assertEqual(nb, builder.build(personal, digest))
            env = {}
            with patch('subprocess.run') as run, patch('subprocess.check_output') as check:
                run.return_value.stdout = 'GPU fictícia'
                run.return_value.returncode = 0
                for cell in nb['cells']:
                    self.assertTrue(cell['id'])
                    if cell['cell_type'] != 'code': continue
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
                    exec(compile(''.join(cell['source']), name, 'exec'), env)
                check.assert_not_called()
                for call in run.call_args_list:
                    self.assertIn('nvidia-smi', call.args[0][0])
            text = ''.join(''.join(c['source']) for c in nb['cells'])
            self.assertNotIn('piloto-auto-001', text)
            self.assertNotIn('run-002-g4', text)

    def test_source_manifest_matches_and_bootstrap_refuses_tampering_before_tests(self):
        manifest = json.loads((ROOT / 'SOURCE-MANIFEST.json').read_text())
        self.assertEqual(manifest, builder.source_manifest())
        nb = json.loads((ROOT / 'COMECE-AQUI.ipynb').read_text())
        source = next(''.join(c['source']) for c in nb['cells']
                      if 'OBTER_CODIGO = False' in ''.join(c['source']))
        source = source.replace('OBTER_CODIGO = False', 'OBTER_CODIGO = True')
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'repo'
            shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns('.git', '__pycache__'))
            (destination / 'ft05/community.py').write_text('raise RuntimeError("tampered")')
            env = {'REPO_DIR': destination, 'Path': Path, 'json': json, 'hashlib': hashlib,
                   'subprocess': __import__('subprocess'), 'sys': sys}
            with patch('subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'alterado'): exec(source, env)
                run.assert_not_called()

    def test_bootstrap_rejects_unlisted_bytecode_before_any_import(self):
        nb = json.loads((ROOT / 'COMECE-AQUI.ipynb').read_text())
        source = next(''.join(c['source']) for c in nb['cells'] if 'OBTER_CODIGO = False' in ''.join(c['source']))
        source = source.replace('OBTER_CODIGO = False', 'OBTER_CODIGO = True')
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'repo'
            shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns('.git', '__pycache__'))
            (destination / 'ft05/injected.pyc').write_bytes(b'not-executed')
            with patch('subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'compilado inesperado'):
                    exec(source, {'REPO_DIR': destination, 'Path': Path, 'sys': sys})
                run.assert_not_called()

    def test_paths_reject_traversal_absolute_paths_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'drive'; root.mkdir()
            for folder in ('/tmp/elsewhere', '../elsewhere', 'x/../y', 'a//b', 'a/./b'):
                with self.subTest(folder=folder), self.assertRaises(ValueError): paths(folder, 'run-001', root)
            with self.assertRaises(ValueError): paths('safe', '../bad', root)
            (root / 'escape').symlink_to(Path(directory))
            with self.assertRaises(ValueError): paths('escape', 'run-001', root)

    def test_dependency_conflict_prevents_gpu_actions(self):
        nb = json.loads((ROOT / 'COMECE-AQUI.ipynb').read_text())
        codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        install = next(c for c in codes if 'INSTALAR_DEPENDENCIAS = False' in c)
        train = next(c for c in codes if c.startswith('TREINAR = False')).replace('TREINAR = False', 'TREINAR = True')
        with tempfile.TemporaryDirectory() as directory:
            env = {'CONFERIR_DEPENDENCIAS': True, 'ROOT': Path(directory), 'sys': sys,
                   'subprocess': __import__('subprocess'), 'MONTAR_DRIVE': True,
                   'PREPARAR_CORPUS': True}
            with patch('community.require_packages'), patch('community.save_environment'), patch('subprocess.run') as run:
                run.return_value.returncode, run.return_value.stdout = 1, 'dependency conflict'
                with self.assertRaisesRegex(ValueError, 'Conflitos'): exec(install, env)
                self.assertFalse(env['AMBIENTE_OK'])
                with self.assertRaisesRegex(ValueError, 'Conclua'): exec(train, env)

    def test_loaded_interface_cannot_silently_switch_runs(self):
        nb = json.loads((ROOT / 'TESTAR-ADAPTADOR.ipynb').read_text())
        code = next(''.join(c['source']) for c in nb['cells'] if 'ABRIR_INTERFACE = False' in ''.join(c['source']))
        code = code.replace('ABRIR_INTERFACE = False', 'ABRIR_INTERFACE = True')
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory); (run_dir / 'run.json').write_text('{}')
            env = {'MONTAR_DRIVE': True, 'AMBIENTE_OK': True, 'RUN_DIR': run_dir,
                   'hashlib': hashlib, 'panel': object(), 'panel_identity': ('old-run', 'old-hash')}
            with patch('community.require_packages'):
                with self.assertRaisesRegex(ValueError, 'Outra execução'): exec(code, env)

    def test_corpus_preparation_retries_validate_without_changing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project'
            first = prepare_project(root)
            before = (root / 'frozen/test.jsonl').read_bytes()
            self.assertEqual(first, prepare_project(root))
            self.assertEqual(first['counts'], {'train': 162, 'validation': 12, 'test': 34})
            self.assertFalse(first['human_reviewed'])
            self.assertEqual(before, (root / 'frozen/test.jsonl').read_bytes())
            (root / 'frozen/test.jsonl').write_text('tampered\n')
            with self.assertRaises(ValueError): prepare_project(root)


if __name__ == '__main__':
    unittest.main()
