"""Check the Colab process boundary without installing or importing GPU packages."""
import builtins
import copy
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import colab_environment as environment
from dependencies import BACKEND, VERSIONS


class ColabEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repository'
        (self.repo / 'ft05').mkdir(parents=True)
        for name in ('ft05/environment-lock.json', 'requirements-colab-lock.txt'):
            shutil.copyfile(environment.ROOT / name, self.repo / name)
        self.lock_file = self.repo / 'ft05/environment-lock.json'
        self.requirements_file = self.repo / 'requirements-colab-lock.txt'
        self.lock = json.loads(self.lock_file.read_text())
        self.requirements = self.requirements_file.read_text()

    def write_lock(self, value):
        self.lock_file.write_text(json.dumps(value), encoding='utf-8')

    def make_environment(self, path=None, config=None):
        path = self.root / 'venv' if path is None else path
        (path / 'bin').mkdir(parents=True)
        (path / 'bin/python').write_text('unused; all subprocesses are mocked')
        (path / environment.MARKER).write_text(json.dumps({'backend': BACKEND}))
        (path / 'pyvenv.cfg').write_text(config or
            'home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.12.13\n')
        return path

    def installed_packages(self):
        return {row['name'].lower().replace('_', '-'): row['version']
                for row in self.lock['distributions']}

    def metadata_result(self, path, packages=None, *, prefix=None, base_prefix='/usr'):
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps({
            'prefix': str(path) if prefix is None else prefix,
            'base_prefix': base_prefix,
            'packages': self.installed_packages() if packages is None else packages}))

    def test_release_lock_has_94_unique_hashed_official_wheels_and_compatible_pins(self):
        lock, digest = environment.locked_plan(self.repo)
        self.assertEqual(len(lock['distributions']), 94)
        self.assertEqual(digest, hashlib.sha256(self.requirements_file.read_bytes()).hexdigest())
        versions = {row['name'].lower().replace('_', '-'): row['version']
                    for row in lock['distributions']}
        self.assertEqual(len(versions), 94)
        self.assertEqual({name: versions[name] for name in VERSIONS}, VERSIONS)
        self.assertEqual(versions['datasets'], '5.0.1')
        self.assertEqual(versions['torch'], '2.13.0+cu129')
        self.assertNotIn('unsloth', versions)
        self.assertNotIn('bitsandbytes', versions)
        self.assertEqual(self.requirements.count('--hash=sha256:'), 94)

    def test_lock_refuses_missing_duplicate_or_changed_direct_dependency(self):
        for mutation, error in (
                ('missing', 'dependency_lock_invalid'),
                ('duplicate', 'dependency_lock_invalid'),
                ('wrong_pin', 'dependency_lock_pins_changed')):
            with self.subTest(mutation=mutation):
                lock = copy.deepcopy(self.lock)
                if mutation == 'missing':
                    lock['distributions'].pop()
                elif mutation == 'duplicate':
                    lock['distributions'][-1] = copy.deepcopy(lock['distributions'][0])
                else:
                    next(row for row in lock['distributions']
                         if row['name'] == 'datasets')['version'] = '4.8.5'
                self.write_lock(lock)
                with self.assertRaisesRegex(ValueError, error):
                    environment.locked_plan(self.repo)

    def test_lock_refuses_changed_python_platform_or_cuda_target(self):
        for key, value in (('python', '3.13'), ('platform', 'linux-aarch64'),
                           ('torch_cuda', '13.0')):
            with self.subTest(key=key):
                lock = copy.deepcopy(self.lock)
                lock[key] = value
                self.write_lock(lock)
                with self.assertRaisesRegex(ValueError, 'dependency_lock_platform_changed'):
                    environment.locked_plan(self.repo)

    def test_lock_refuses_untrusted_urls_or_invalid_hash_even_if_requirements_agree(self):
        original = self.lock['distributions'][0]
        unsafe_urls = (
            original['url'].replace('https:', 'http:'),
            original['url'].replace('files.pythonhosted.org', 'packages.example.org'),
            original['url'].replace('files.pythonhosted.org', 'files.pythonhosted.org.example.org'),
            original['url'].replace('https://', 'https://user:secret@'),
            original['url'] + '?redirect=example.org',
            original['url'] + '#sha256=' + original['sha256'],
            original['url'].replace('.whl', '.tar.gz'),
        )
        mutations = [('url', url) for url in unsafe_urls]
        mutations += [('sha256', 'a' * 63), ('sha256', 'g' * 64), ('sha256', 'A' * 64)]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                lock = copy.deepcopy(self.lock)
                lock['distributions'][0][field] = value
                self.write_lock(lock)
                self.requirements_file.write_text(
                    self.requirements.replace(original[field], value), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'dependency_wheel_source_invalid'):
                    environment.locked_plan(self.repo)

    def test_lock_and_requirements_tampering_are_rejected_before_installation(self):
        lock = copy.deepcopy(self.lock)
        row = lock['distributions'][0]
        row['sha256'] = ('0' if row['sha256'][0] != '0' else '1') + row['sha256'][1:]
        self.write_lock(lock)
        with patch.object(environment, 'check_platform'), \
                patch.object(environment.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'dependency_requirements_changed'):
                environment.install_environment(self.root / 'fresh', self.repo)
            run.assert_not_called()
        self.write_lock(self.lock)
        self.requirements_file.write_text(self.requirements + '\n--extra-index-url https://example.org\n')
        with self.assertRaisesRegex(ValueError, 'dependency_requirements_changed'):
            environment.locked_plan(self.repo)

    def test_venv_refuses_system_packages_wrong_marker_and_commented_false(self):
        path = self.make_environment()
        configs = (
            'include-system-site-packages = true\n',
            '# include-system-site-packages = false\ninclude-system-site-packages = true\n',
            'include-system-site-packages = false\ninclude-system-site-packages = true\n',
        )
        for config in configs:
            with self.subTest(config=config):
                (path / 'pyvenv.cfg').write_text(config)
                with patch.object(environment.subprocess, 'run') as run:
                    with self.assertRaises(ValueError):
                        environment.environment_status(path)
                    run.assert_not_called()
        (path / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        (path / environment.MARKER).write_text(json.dumps({'backend': 'another-project'}))
        with patch.object(environment, 'check_platform'), \
                patch.object(environment.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                environment.install_environment(path, self.repo)
            run.assert_not_called()

    def test_venv_refuses_direct_parent_and_dangling_symlink_paths(self):
        actual = self.make_environment(self.root / 'actual' / 'venv')
        direct = self.root / 'direct'
        direct.symlink_to(actual, target_is_directory=True)
        parent = self.root / 'parent'
        parent.symlink_to(actual.parent, target_is_directory=True)
        dangling = self.root / 'dangling'
        dangling.symlink_to(self.root / 'absent', target_is_directory=True)
        for alias in (direct, parent / 'venv', dangling):
            with self.subTest(alias=alias), \
                    patch.object(environment, 'check_platform'), \
                    patch.object(environment.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'environment_path_alias'):
                    environment.install_environment(alias, self.repo)
                run.assert_not_called()

    def test_environment_status_rejects_global_or_wrong_interpreter_prefix(self):
        path = self.make_environment()
        for prefix, base in (('/usr', '/usr'), (str(self.root / 'other'), '/usr'),
                             (str(path), str(path))):
            with self.subTest(prefix=prefix, base=base), \
                    patch.object(environment.subprocess, 'run', return_value=
                        self.metadata_result(path, prefix=prefix, base_prefix=base)):
                with self.assertRaisesRegex(ValueError, 'environment_not_isolated'):
                    environment.environment_status(path)

    def test_missing_or_wrong_versions_stop_before_pip_check_or_cuda_import(self):
        path = self.make_environment()
        for version in (None, '4.8.5'):
            packages = self.installed_packages()
            if version is None:
                packages.pop('datasets')
            else:
                packages['datasets'] = version
            with self.subTest(version=version), \
                    patch.object(environment, 'check_platform'), \
                    patch.object(environment.subprocess, 'run', return_value=
                        self.metadata_result(path, packages)) as run:
                with self.assertRaisesRegex(ValueError, 'datasets'):
                    environment.validate_environment(path, self.repo)
                run.assert_called_once()
                self.assertIn('importlib.metadata', run.call_args.args[0][-1])

    def test_pip_conflicts_stop_before_cuda_process(self):
        path = self.make_environment()
        with patch.object(environment, 'check_platform'), \
                patch.object(environment.subprocess, 'run', side_effect=[
                    self.metadata_result(path),
                    SimpleNamespace(returncode=1, stdout='dependency conflict', stderr=''),
                ]) as run:
            with self.assertRaisesRegex(ValueError, 'Conflitos no ambiente isolado'):
                environment.validate_environment(path, self.repo)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args[0],
                [sys.executable, '-I', '-m', 'pip', '--python',
                 str(path / 'bin/python'), '--disable-pip-version-check', 'check'])

    def test_install_only_creates_venv_with_base_python_and_installs_in_venv(self):
        path = self.root / 'fresh'

        def fake_process(command, **kwargs):
            if command[:4] == [sys.executable, '-I', '-m', 'venv']:
                self.make_environment(path)
            return SimpleNamespace(returncode=0)

        with patch.object(environment, 'check_platform'), \
                patch.object(environment.subprocess, 'run', side_effect=fake_process) as run:
            environment.install_environment(path, self.repo)
        self.assertEqual(run.call_count, 2)
        create, install = [call.args[0] for call in run.call_args_list]
        self.assertEqual(create,
            [sys.executable, '-I', '-m', 'venv', '--without-pip', str(path)])
        self.assertEqual(install[:8],
            [sys.executable, '-I', '-m', 'pip', '--python',
             str(path / 'bin/python'), '--disable-pip-version-check', 'install'])
        for flag in ('--require-hashes', '--only-binary=:all:', '--no-index'):
            self.assertIn(flag, install)
        self.assertNotIn('--no-deps', install)
        self.assertNotIn('--system-site-packages', create)
        self.assertEqual(install[-1], str(self.requirements_file))
        self.assertEqual(json.loads((path / environment.MARKER).read_text()), {'backend': BACKEND})

    def test_validated_cuda_runs_only_in_isolated_child_and_never_imports_gpu_stack_here(self):
        path = self.make_environment()
        original_import = builtins.__import__

        def reject_gpu_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'trl', 'datasets', 'peft', 'accelerate'}:
                raise AssertionError('GPU package imported in notebook kernel: ' + name)
            return original_import(name, *args, **kwargs)

        gpu = {'name': 'mock Colab GPU', 'device': 'cuda:0', 'precision': 'bf16'}
        with patch.object(environment, 'check_platform'), \
                patch.object(environment.subprocess, 'run', side_effect=[
                    self.metadata_result(path),
                    SimpleNamespace(returncode=0, stdout='No broken requirements found.', stderr=''),
                    SimpleNamespace(returncode=0, stdout=json.dumps(gpu), stderr=''),
                ]) as run, patch('builtins.__import__', side_effect=reject_gpu_import):
            summary = environment.validate_environment(path, self.repo)
        self.assertEqual(summary['gpu'], gpu)
        self.assertEqual(summary['training_backend'], BACKEND)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            self.assertIn('-I', call.args[0])
        self.assertEqual(run.call_args_list[0].args[0][0], str(path / 'bin/python'))
        self.assertEqual(run.call_args_list[1].args[0][0], sys.executable)
        self.assertEqual(run.call_args_list[1].args[0][4:6],
                         ['--python', str(path / 'bin/python')])
        self.assertEqual(run.call_args_list[2].args[0][0], str(path / 'bin/python'))
        self.assertIn('require_packages();', run.call_args.args[0][-2])
        self.assertIn('cuda_runtime()', run.call_args.args[0][-2])
        self.assertEqual(run.call_args.args[0][-1], str(self.repo / 'ft05'))

    def test_existing_unmarked_partial_environment_is_not_repurposed(self):
        path = self.make_environment()
        (path / environment.MARKER).unlink()
        with patch.object(environment, 'check_platform'), \
                patch.object(environment.subprocess, 'run') as run:
            with self.assertRaises(FileNotFoundError):
                environment.install_environment(path, self.repo)
            run.assert_not_called()
        self.assertTrue(path.is_dir())

    def test_real_venv_without_pip_accepts_kernel_pip_target_check_without_network(self):
        path = self.root / 'real-without-pip'
        kernel_before = sorted((row.metadata['Name'], row.version)
                               for row in importlib.metadata.distributions()
                               if row.metadata['Name'])
        subprocess.run([sys.executable, '-I', '-m', 'venv', '--without-pip', str(path)],
                       check=True, capture_output=True, text=True, timeout=60)
        (path / environment.MARKER).write_text(json.dumps({'backend': BACKEND}))

        code = ('import importlib.util,json,sys; '
                'print(json.dumps({"prefix":sys.prefix,"base_prefix":sys.base_prefix,'
                '"pip":importlib.util.find_spec("pip") is not None,'
                '"torch":importlib.util.find_spec("torch") is not None}))')
        before = subprocess.run([str(path / 'bin/python'), '-I', '-c', code],
                                check=True, capture_output=True, text=True, timeout=30)
        state = json.loads(before.stdout)
        self.assertEqual(Path(state['prefix']).resolve(), path.resolve())
        self.assertNotEqual(state['prefix'], state['base_prefix'])
        self.assertFalse(state['pip'])
        self.assertFalse(state['torch'])
        self.assertEqual(environment.environment_status(path), {})

        result = subprocess.run(environment._pip_command(path / 'bin/python') + ['check'],
                                check=True, capture_output=True, text=True, timeout=60)
        self.assertIn('No broken requirements found.', result.stdout)
        self.assertEqual(environment.environment_status(path), {})
        after = subprocess.run([str(path / 'bin/python'), '-I', '-c', code],
                               check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(after.stdout), state)
        kernel_after = sorted((row.metadata['Name'], row.version)
                              for row in importlib.metadata.distributions()
                              if row.metadata['Name'])
        self.assertEqual(kernel_after, kernel_before)

    def test_validation_without_cuda_does_not_start_gpu_process(self):
        path = self.make_environment()
        with patch.object(environment, 'check_platform'), \
                patch.object(environment.subprocess, 'run', side_effect=[
                    self.metadata_result(path),
                    SimpleNamespace(returncode=0, stdout='', stderr=''),
                ]) as run:
            summary = environment.validate_environment(path, self.repo, check_cuda=False)
        self.assertNotIn('gpu', summary)
        self.assertEqual(run.call_count, 2)

    def test_cuda_driver_or_kernel_failure_prevents_a_ready_summary(self):
        path = self.make_environment()
        failure = subprocess.CalledProcessError(1, ['isolated-python', 'cuda-probe'],
                                               stderr='CUDA kernel is not supported')
        with patch.object(environment, 'check_platform'), \
                patch.object(environment.subprocess, 'run', side_effect=[
                    self.metadata_result(path),
                    SimpleNamespace(returncode=0, stdout='', stderr=''),
                    failure,
                ]) as run:
            with self.assertRaisesRegex(ValueError, 'CUDA kernel is not supported'):
                environment.validate_environment(path, self.repo)
        probe = run.call_args.args[0][-2]
        self.assertIn('torch.ones', probe)
        self.assertIn('torch.cuda.synchronize()', probe)
        self.assertIs(run.call_args.kwargs['check'], True)

    def test_unsupported_platform_stops_before_any_installation(self):
        for version, system, machine in (((3, 13), 'linux', 'x86_64'),
                                         ((3, 12), 'win32', 'x86_64'),
                                         ((3, 12), 'linux', 'aarch64')):
            with self.subTest(version=version, system=system, machine=machine), \
                    patch.object(environment.sys, 'version_info', version), \
                    patch.object(environment.sys, 'platform', system), \
                    patch.object(environment.platform, 'machine', return_value=machine), \
                    patch.object(environment.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'Python 3.12, Linux x86_64'):
                    environment.install_environment(self.root / 'fresh', self.repo)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
