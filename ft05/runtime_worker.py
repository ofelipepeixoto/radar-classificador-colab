"""Private JSONL inference worker; run only with the isolated runtime Python.

The notebook kernel never imports the model stack. Only labels and latency
leave this process; input text, model output, and library logs are not returned.
"""
import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

PROTOCOL = 1
MAX_LINE_BYTES = 65536


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_key')
            result[key] = value
        return result

    def invalid(_):
        raise ValueError('nonfinite_json')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def serve(input_stream, protocol_stream, run_dir, corpus, frozen, *, backend=None):
    """Serve one model and sequential messages; injectable backend for CPU tests."""
    def emit(value):
        protocol_stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        protocol_stream.flush()

    try:
        # Redirect Python prints too, including when serve is used by CPU tests.
        with contextlib.redirect_stdout(sys.stderr):
            if backend is None:
                from personal_smoke import preflight, load_saved_adapter, validate_messages
                from train import generate_response
                from ft05 import target
            else:
                preflight, load_saved_adapter, validate_messages, generate_response, target = backend
            checked = preflight(run_dir, corpus, frozen, versions=True)
            model, tokenizer, _ = load_saved_adapter(checked)
    except Exception:
        emit({'error': 'startup_failed'})
        return 2

    emit({'ready': True, 'protocol': PROTOCOL})
    while True:
        line = input_stream.readline(MAX_LINE_BYTES + 1)
        if not line:
            return 0
        # Oversized/incomplete frames cannot be safely resynchronized.
        if len(line.encode('utf-8')) > MAX_LINE_BYTES or not line.endswith('\n'):
            emit({'error': 'invalid_request'})
            return 2
        ident = None
        try:
            request = _strict_json(line)
            if type(request) is not dict or set(request) != {'id', 'text'}:
                raise ValueError('invalid_request')
            with contextlib.redirect_stdout(sys.stderr):
                validate_messages([request])
            ident = request['id']
        except (ValueError, TypeError, KeyError):
            emit({'error': 'invalid_request'})
            # Fail closed; do not return arbitrary unvalidated request IDs.
            return 2
        try:
            with contextlib.redirect_stdout(sys.stderr):
                raw, elapsed_ms = generate_response(
                    model, tokenizer, [{'role': 'user', 'content': request['text']}])
                labels = target(raw)
            if (type(elapsed_ms) not in (int, float)
                    or not math.isfinite(elapsed_ms) or elapsed_ms < 0):
                raise ValueError('invalid_latency')
            emit({'id': ident, 'labels': labels, 'latency_ms': elapsed_ms})
        except Exception:
            emit({'id': ident, 'error': 'inference_failed'})
            return 2


def main():
    # Capture the protocol FD before redirecting fd1 itself. This catches native
    # libraries writing to stdout, beyond contextlib.redirect_stdout.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    parser = argparse.ArgumentParser(description='Private Colab inference worker')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--frozen', required=True)
    args = parser.parse_args()
    # -I excludes cwd/PYTHONPATH. Only this checked source directory is needed.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        return serve(sys.stdin, protocol, args.run_dir, args.corpus, args.frozen)
    except (BrokenPipeError, OSError, UnicodeError):
        return 2
    finally:
        protocol.close()


if __name__ == '__main__':
    raise SystemExit(main())
