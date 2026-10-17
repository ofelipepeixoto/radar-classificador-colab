"""Browser widgets backed by one isolated subprocess; no model imports here."""
import atexit
import json
import math
import queue
import re
import subprocess
import threading
import uuid
from pathlib import Path

MAX_FRAME_BYTES = 16384
_ERRORS = {
    'startup_failed': 'Não foi possível verificar ou carregar o adaptador. Confira o ambiente e os arquivos nas etapas anteriores.',
    'invalid_request': 'Mensagem rejeitada. Use texto fictício, sem dados pessoais ou segredos.',
    'inference_failed': 'Não foi possível classificar a mensagem com o formato esperado. Reabra a interface para tentar novamente.',
}


class WorkerError(RuntimeError):
    """An intentionally sanitized failure safe to show in a notebook output."""


def _strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate_key')
            value[key] = item
        return value

    def invalid(_):
        raise ValueError('nonfinite_json')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


class RuntimeWorker:
    """One model bound to immutable paths; kill on timeout, EOF or bad protocol."""
    def __init__(self, runtime_python, run_dir, corpus, frozen, *,
                 ready_timeout=180, request_timeout=60, worker_path=None):
        if ready_timeout <= 0 or request_timeout <= 0:
            raise ValueError('timeouts_must_be_positive')
        # A venv Python is commonly a symlink to the base interpreter. Resolving
        # that symlink would silently escape the venv and its pinned packages.
        self.identity = (str(Path(runtime_python).absolute()),) + tuple(
            str(Path(path).resolve()) for path in (run_dir, corpus, frozen))
        self.request_timeout = request_timeout
        self._events = queue.Queue(maxsize=4)
        self._stream_failed = threading.Event()
        self._closed = False
        self._close_requested = threading.Event()
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        source = Path(worker_path) if worker_path is not None else Path(__file__).with_name('runtime_worker.py')
        command = [self.identity[0], '-I', '-B', '-u', str(source.resolve()),
                   '--run-dir', self.identity[1], '--corpus', self.identity[2], '--frozen', self.identity[3]]
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                bufsize=1, shell=False)
        except (OSError, ValueError):
            self._closed = True
            raise WorkerError('Não foi possível iniciar o Python isolado. Repita a instalação do ambiente.') from None
        atexit.register(self.close)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        threading.Thread(target=self._reap_when_closed, daemon=True).start()
        try:
            ready = self._receive(ready_timeout)
            if (ready != {'ready': True, 'protocol': 1}
                    or type(ready.get('ready')) is not bool or type(ready.get('protocol')) is not int):
                self._raise_protocol(ready)
        except BaseException:
            self.close()
            raise

    @property
    def closed(self):
        return self._closed

    def _put(self, event):
        try:
            self._events.put_nowait(event)
        except queue.Full:
            self._stream_failed.set()
            self.close()

    def _read_stdout(self):
        try:
            while not self._closed:
                line = self._process.stdout.readline(MAX_FRAME_BYTES + 1)
                if not line:
                    self._put(('eof', None))
                    return
                if len(line.encode('utf-8')) > MAX_FRAME_BYTES or not line.endswith('\n'):
                    self._stream_failed.set()
                    self._put(('invalid', None))
                    return
                self._put(('frame', line))
        except (OSError, UnicodeError, ValueError):
            self._stream_failed.set()
            self._put(('invalid', None))

    def _drain_stderr(self):
        # Never retain or display library diagnostics: they may contain input,
        # paths or credentials. Continuous draining avoids pipe deadlocks.
        try:
            while self._process.stderr.read(4096):
                pass
        except (OSError, UnicodeError, ValueError):
            pass

    def _raise_protocol(self, frame=None):
        code = frame.get('error') if type(frame) is dict else None
        raise WorkerError(_ERRORS.get(code, 'Comunicação inválida com o ambiente isolado. Reabra a interface.'))

    def _receive(self, timeout):
        if self._stream_failed.is_set():
            self._raise_protocol()
        try:
            kind, value = self._events.get(timeout=timeout)
        except queue.Empty:
            raise WorkerError('O ambiente isolado excedeu o tempo de espera. Reabra a interface para tentar novamente.') from None
        if self._stream_failed.is_set() or kind == 'invalid':
            self._raise_protocol()
        if kind == 'eof':
            raise WorkerError('O ambiente isolado foi encerrado. Reabra a interface.')
        try:
            frame = _strict_json(value)
        except (TypeError, ValueError):
            self._raise_protocol()
        if type(frame) is not dict:
            self._raise_protocol()
        return frame

    def classify(self, text):
        if type(text) is not str or not 3 <= len(text.strip()) <= 2000:
            raise WorkerError('Digite entre 3 e 2.000 caracteres.')
        if re.search(r'\S+@\S+|\b\d[\d .()+-]{8,}\d\b|sk-[A-Za-z0-9]{12,}', text):
            raise WorkerError(_ERRORS['invalid_request'])
        if self._closed:
            raise WorkerError('A interface está encerrada. Execute sua célula novamente.')
        if not self._request_lock.acquire(blocking=False):
            raise WorkerError('Aguarde a classificação atual terminar.')
        try:
            ident = uuid.uuid4().hex
            self._process.stdin.write(json.dumps({'id': ident, 'text': text.strip()}, ensure_ascii=False) + '\n')
            self._process.stdin.flush()
            frame = self._receive(self.request_timeout)
            if set(frame) != {'id', 'labels', 'latency_ms'} or frame.get('id') != ident:
                self._raise_protocol(frame)
            labels, latency = frame['labels'], frame['latency_ms']
            if (type(labels) is not dict or set(labels) != {'categoria', 'urgencia'}
                    or type(labels['categoria']) is not str or type(labels['urgencia']) is not str
                    or labels['categoria'] not in {'entrega', 'pagamento', 'cancelamento', 'outro'}
                    or labels['urgencia'] not in {'alta', 'media', 'baixa'}
                    or type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
                self._raise_protocol()
            return frame
        except WorkerError:
            self.close()
            raise
        except (OSError, ValueError, UnicodeError):
            self.close()
            raise WorkerError('O ambiente isolado foi encerrado. Reabra a interface.') from None
        except BaseException:
            self.close()
            raise
        finally:
            self._request_lock.release()

    def close(self):
        """Signal termination immediately; bounded reaping happens off the UI thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        atexit.unregister(self.close)
        try:
            self._process.terminate()
        except OSError:
            pass
        self._close_requested.set()

    def _reap_when_closed(self):
        # Start this thread with the process, not from close(): Python 3.12+
        # forbids spawning new threads inside an atexit handler.
        self._close_requested.wait()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                self._process.kill()
                self._process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        finally:
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass


def launch(runtime_python, run_dir, corpus, frozen, *, ready_timeout=180, request_timeout=60):
    """Render text input and Classificar; one model per explicitly selected run."""
    import ipywidgets as widgets
    from IPython.display import display

    worker = RuntimeWorker(runtime_python, run_dir, corpus, frozen,
                           ready_timeout=ready_timeout, request_timeout=request_timeout)
    box = widgets.Textarea(value='', placeholder='Digite uma mensagem fictícia...',
                           description='Mensagem:', layout=widgets.Layout(width='95%', height='110px'))
    button = widgets.Button(description='Classificar', button_style='primary')
    stop = widgets.Button(description='Encerrar interface')
    output = widgets.Output()

    def classify(_):
        button.disabled = True
        try:
            with output:
                output.clear_output()
                result = worker.classify(box.value)
                print('Categoria:', result['labels']['categoria'])
                print('Urgência:', result['labels']['urgencia'])
                print('Tempo:', round(result['latency_ms']), 'ms')
                print('Teste pessoal sintético; não é decisão automática para clientes.')
        except WorkerError as exc:
            with output:
                print(str(exc))
        finally:
            button.disabled = worker.closed

    def stop_worker(_):
        worker.close()
        button.disabled = True
        stop.disabled = True
        box.disabled = True
        with output:
            output.clear_output()
            print('Interface encerrada. Execute a célula novamente para reabrir.')

    class WorkerPanel(widgets.VBox):
        def close(self):
            worker.close()
            super().close()

    button.on_click(classify)
    stop.on_click(stop_worker)
    panel = WorkerPanel([box, widgets.HBox([button, stop]), output])
    panel._ft05_worker = worker
    display(panel)
    print('Interface pronta. A GPU é usada pelo ambiente isolado do Colab.')
    return panel
