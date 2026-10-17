"""Fail on accidental weights, common secrets or saved notebook outputs. Offline.

This is a narrow publication tripwire, not a full secret scanner or SCA audit.
"""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = {'.safetensors', '.bin', '.gguf', '.pt', '.pth', '.pem', '.key'}
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')


def check():
    try:
        names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
        files = [ROOT / name for name in names]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts
                 and '__pycache__' not in p.parts]
    manifest = json.loads((ROOT / 'SOURCE-MANIFEST.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source manifest changed: ' + name)
    for file in files:
        if file.suffix in WEIGHT_SUFFIXES or file.name.startswith('.env'):
            raise ValueError('Private/weight file included: ' + str(file.relative_to(ROOT)))
        text = file.read_text(encoding='utf-8')
        if SECRET.search(text):
            raise ValueError('Possible secret; inspect locally: ' + str(file.relative_to(ROOT)))
        if file.suffix == '.py': ast.parse(text)
        if file.suffix == '.ipynb':
            nb = json.loads(text)
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    if cell['outputs'] or cell['execution_count'] is not None:
                        raise ValueError('Notebook contains saved outputs: ' + file.name)
                    ast.parse(''.join(cell['source']))
    print(f'Publication checks: OK ({len(files)} text files; no weights, common-secret matches or saved outputs).')


if __name__ == '__main__':
    check()
