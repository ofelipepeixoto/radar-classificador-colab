"""CPU-only tests of the subprocess boundary; no model imports or downloads."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import colab_ui
from colab_ui import RuntimeWorker, WorkerError
from runtime_worker import serve
from personal_smoke import validate_messages
from ft05 import target


class WorkerProtocolTest(unittest.TestCase):
    def backend(self, *, invalid=False, fail=False):
        self.loads, self.generations, self.preflights = [], [], []

        def preflight(*args, **kwargs):
            self.preflights.append((args, kwargs))
            print('private preflight log')
            return {'checked': True}

        def load(checked):
            self.loads.append(checked)
            print('private load log')
            return 'one-model', 'one-tokenizer', 'base'

        def generate(model, tokenizer, messages):
            self.generations.append((model, tokenizer, messages))
            print('private generation log')
            if fail:
                raise ValueError('secret input must never escape')
            return ('raw secret' if invalid else '{"categoria":"entrega","urgencia":"media"}', 12.5)

        return preflight, load, validate_messages, generate, target

    def run_server(self, messages, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            code = serve(io.StringIO(messages), output, 'run', 'corpus', 'frozen', backend=self.backend(**kwargs))
        return code, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_one_load_two_messages_labels_only(self):
        code, frames = self.run_server(
            '{"id":"a","text":"Texto fictício de entrega"}\n'
            '{"id":"b","text":"Outro texto fictício de entrega"}\n')
        self.assertEqual(code, 0)
        self.assertEqual(frames[0], {'ready': True, 'protocol': 1})
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(len(self.generations), 2)
        self.assertEqual(self.preflights[0][1], {'versions': True})
        self.assertEqual(set(frames[1]), {'id', 'labels', 'latency_ms'})
        self.assertEqual(frames[2]['id'], 'b')
        self.assertNotIn('text', frames[1])

    def test_sensitive_message_rejected_before_generation(self):
        code, frames = self.run_server('{"id":"a","text":"Contato: pessoa@example.com"}\n')
        self.assertEqual(code, 2)
        self.assertEqual(frames[-1], {'error': 'invalid_request'})
        self.assertEqual(self.generations, [])

    def test_duplicate_keys_incomplete_and_oversized_request_rejected(self):
        for value in ('{"id":"a","id":"b","text":"abc"}\n',
                      '{"id":"a","text":"abc"}', 'x' * 65537 + '\n'):
            with self.subTest(value=value[:50]):
                code, frames = self.run_server(value)
                self.assertEqual(code, 2)
                self.assertEqual(frames[-1], {'error': 'invalid_request'})
                self.assertEqual(self.generations, [])

    def test_inference_and_target_errors_do_not_echo(self):
        for options in ({'fail': True}, {'invalid': True}):
            with self.subTest(options=options):
                code, frames = self.run_server('{"id":"a","text":"texto fictício"}\n', **options)
                self.assertEqual(code, 2)
                self.assertEqual(frames[-1], {'id': 'a', 'error': 'inference_failed'})

    def test_startup_failure_has_constant_message(self):
        output = io.StringIO()

        def failed(*args, **kwargs):
            raise ValueError('secret credential')

        self.assertEqual(serve(io.StringIO(), output, 'r', 'c', 'f',
                               backend=(failed, None, None, None, None)), 2)
        self.assertEqual(json.loads(output.getvalue()), {'error': 'startup_failed'})

    def test_main_redirects_python_and_native_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath('runtime_worker.py').write_text(
                Path(__file__).resolve().parents[1].joinpath('runtime_worker.py').read_text())
            root.joinpath('personal_smoke.py').write_text('''
import os
print('private import log')
os.write(1, b'private native log\\n')
def preflight(*args, **kwargs): return {}
def load_saved_adapter(checked): return object(), object(), 'base'
def validate_messages(rows): return rows
''')
            root.joinpath('train.py').write_text('''
def generate_response(*args):
    print('private generation log')
    return '{"categoria":"entrega","urgencia":"media"}', 3.0
''')
            root.joinpath('ft05.py').write_text('import json\ndef target(value): return json.loads(value)\n')
            result = subprocess.run([sys.executable, '-I', '-u', str(root / 'runtime_worker.py'),
                                     '--run-dir', 'r', '--corpus', 'c', '--frozen', 'f'],
                                    input='{"id":"a","text":"abc"}\n', text=True,
                                    capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0)
            frames = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(frames), 2)
            self.assertNotIn('private', result.stdout)
            self.assertIn('private native log', result.stderr)


class ClientSubprocessTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def worker(self, mode='ok', *, ready_timeout=2, request_timeout=0.3, runtime_python=None):
        source = self.root / ('child_' + mode + '.py')
        source.write_text('''
import json, os, sys, time
mode = MODE
if mode == 'startup_timeout': time.sleep(10)
if mode == 'startup_eof': raise SystemExit(0)
if mode == 'startup_error':
    print(json.dumps({'error': 'startup_failed'}), flush=True)
    raise SystemExit(2)
print(json.dumps({'ready': True, 'protocol': 1}), flush=True)
if mode == 'stderr': os.write(2, b'private secret' * 20000 + b'\\xff')
for line in sys.stdin:
    request = json.loads(line)
    if mode == 'request_timeout': time.sleep(10)
    if mode == 'request_eof': raise SystemExit(0)
    if mode == 'invalid_json':
        print('secret raw output', flush=True)
        continue
    if mode == 'oversized':
        print('x' * 20000, flush=True)
        continue
    if mode == 'duplicate':
        print('{"id":"x","id":"y"}', flush=True)
        continue
    if mode == 'error':
        print(json.dumps({'id': request['id'], 'error': 'inference_failed'}), flush=True)
        continue
    result = {'id': request['id'], 'labels': {'categoria':'entrega','urgencia':'media'},'latency_ms':5.0}
    if mode == 'wrong_id': result['id'] = 'wrong'
    if mode == 'wrong_label': result['labels']['urgencia'] = 'secret'
    if mode == 'nonfinite': result['latency_ms'] = float('nan')
    if mode == 'extra': result['text'] = 'secret'
    print(json.dumps(result), flush=True)
'''.replace('MODE', repr(mode)))
        child = RuntimeWorker(runtime_python or sys.executable, self.root / 'run', self.root / 'corpus', self.root / 'frozen',
                              worker_path=source, ready_timeout=ready_timeout, request_timeout=request_timeout)
        self.addCleanup(child.close)
        return child

    def test_one_process_reused_and_paths_bound(self):
        child = self.worker()
        process = child._process
        self.assertEqual(child.classify('Mensagem fictícia')['labels']['urgencia'], 'media')
        self.assertEqual(child.classify('Outra mensagem fictícia')['labels']['categoria'], 'entrega')
        self.assertIs(child._process, process)
        self.assertEqual(child.identity[1], str((self.root / 'run').resolve()))
        self.assertNotIn('shell=True', Path(colab_ui.__file__).read_text())

    def test_venv_python_symlink_is_preserved(self):
        environment = self.root / 'isolated'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        executable = environment / 'bin' / 'python'
        self.assertTrue(executable.is_symlink())
        child = self.worker(runtime_python=executable)
        self.assertEqual(child._process.args[0], str(executable.absolute()))
        self.assertNotEqual(child._process.args[0], str(executable.resolve()))
        self.assertEqual(child.classify('texto fictício')['labels']['urgencia'], 'media')

    def test_cancelling_request_closes_the_child(self):
        child = self.worker()
        with patch.object(child, '_receive', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                child.classify('texto fictício')
        self.assertTrue(child.closed)
        child._process.wait(timeout=3)

    def test_stderr_is_drained_without_exposing_its_content(self):
        child = self.worker('stderr', request_timeout=2)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            self.assertEqual(child.classify('Texto fictício')['latency_ms'], 5.0)
        self.assertEqual(output.getvalue(), '')

    def test_bad_frames_eof_and_request_timeout_close_process(self):
        for mode in ('invalid_json', 'duplicate', 'wrong_id', 'wrong_label', 'nonfinite',
                     'extra', 'error', 'oversized', 'request_eof', 'request_timeout'):
            with self.subTest(mode=mode):
                child = self.worker(mode)
                with self.assertRaises(WorkerError) as caught:
                    child.classify('Texto fictício')
                self.assertTrue(child.closed)
                self.assertNotIn('secret', str(caught.exception))
                child._process.wait(timeout=3)

    def test_startup_timeout_eof_and_error_are_safe(self):
        for mode in ('startup_timeout', 'startup_eof', 'startup_error'):
            with self.subTest(mode=mode), self.assertRaises(WorkerError) as caught:
                self.worker(mode, ready_timeout=0.3)
            self.assertNotIn('secret', str(caught.exception))

    def test_local_validation_does_not_end_a_healthy_process(self):
        child = self.worker()
        for text in ('', 'x', 'x' * 2001, 'nome@example.com', 'CPF 12345678901'):
            with self.subTest(text=text), self.assertRaises(WorkerError):
                child.classify(text)
            self.assertFalse(child.closed)
        self.assertEqual(child.classify('texto fictício')['labels']['urgencia'], 'media')

    def test_close_is_idempotent_and_cannot_classify_afterwards(self):
        child = self.worker()
        child.close()
        child.close()
        self.assertTrue(child.closed)
        with self.assertRaises(WorkerError):
            child.classify('texto fictício')
        child._process.wait(timeout=3)

    def test_kernel_module_has_no_model_stack_imports(self):
        result = subprocess.run([sys.executable, '-I', '-c',
            "import sys; sys.path.insert(0, " + repr(str(Path(colab_ui.__file__).parent)) + "); "
            "import colab_ui; assert not ({'torch','transformers','peft','personal_smoke','train'} & set(sys.modules))"],
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)


class WidgetsClientTest(unittest.TestCase):
    def test_callback_and_panel_close_use_the_same_worker(self):
        class Widget:
            def __init__(self, children=None, **kwargs):
                self.children = children
                self.disabled = False
                self.__dict__.update(kwargs)

            def on_click(self, callback):
                self.callback = callback

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def clear_output(self):
                pass

            def close(self):
                pass

        class WorkerDouble:
            closed = False
            calls = []

            def __init__(self, *args, **kwargs):
                self.args = args

            def classify(self, text):
                self.calls.append(text)
                return {'labels': {'categoria': 'entrega', 'urgencia': 'alta'}, 'latency_ms': 1.0}

            def close(self):
                self.closed = True

        widgets = SimpleNamespace(Textarea=Widget, Button=Widget, Output=Widget,
                                  VBox=Widget, HBox=Widget, Layout=lambda **kwargs: kwargs)
        modules = {'ipywidgets': widgets, 'IPython.display': SimpleNamespace(display=lambda panel: None)}
        with patch.dict(sys.modules, modules), patch('colab_ui.RuntimeWorker', WorkerDouble), \
                contextlib.redirect_stdout(io.StringIO()):
            panel = colab_ui.launch('python', 'run', 'corpus', 'frozen')
            box, controls, _ = panel.children
            button, stop = controls.children
            box.value = 'mensagem fictícia'
            button.callback(None)
            self.assertEqual(panel._ft05_worker.calls, ['mensagem fictícia'])
            self.assertFalse(button.disabled)
            stop.callback(None)
            self.assertTrue(panel._ft05_worker.closed)
            self.assertTrue(button.disabled)
            panel.close()


if __name__ == '__main__':
    unittest.main()
