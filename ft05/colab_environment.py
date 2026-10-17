"""Hash-locked Colab venv. The notebook kernel never imports the GPU stack."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dependencies import BACKEND, VERSIONS

ROOT = Path(__file__).resolve().parents[1]
MARKER = '.radar-classificador-environment.json'


def locked_plan(repo=ROOT):
    repo = Path(repo)
    lock = json.loads((repo / 'ft05/environment-lock.json').read_text())
    if (lock.get('python'), lock.get('platform'), lock.get('torch_cuda')) != ('3.12', 'linux-x86_64', '12.9'):
        raise ValueError('dependency_lock_platform_changed')
    entries = lock['distributions']
    names = [i['name'].lower().replace('_', '-') for i in entries]
    if len(entries) != 94 or len(set(names)) != len(names):
        raise ValueError('dependency_lock_invalid')
    for item in entries:
        parsed = urlsplit(item['url'])
        if (parsed.scheme != 'https' or parsed.hostname not in
                {'files.pythonhosted.org', 'download-r2.pytorch.org'}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or not parsed.path.endswith('.whl')
                or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('dependency_wheel_source_invalid')
    found = {i['name'].lower().replace('_', '-'): i['version'] for i in entries}
    if any(found.get(name) != version for name, version in VERSIONS.items()):
        raise ValueError('dependency_lock_pins_changed')
    expected = ('# Generated from ft05/environment-lock.json. CPython 3.12, Linux x86_64.\n'
                '# Official wheel URLs, all dependencies fixed; install in the project venv only.\n')
    expected += ''.join(f"{i['name']} @ {i['url']} --hash=sha256:{i['sha256']}\n" for i in entries)
    file = repo / 'requirements-colab-lock.txt'
    if file.read_text() != expected:
        raise ValueError('dependency_requirements_changed')
    return lock, hashlib.sha256(file.read_bytes()).hexdigest()


def check_platform():
    if sys.version_info[:2] != (3, 12) or sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise ValueError('Esta receita requer Python 3.12, Linux x86_64 no Colab.')


def runtime_python(environment):
    return Path(environment) / 'bin/python'


def _pip_command(executable):
    """Use the kernel's pip to manage only the explicit, isolated interpreter.

    Colab may omit ensurepip. pip's --python option supports a venv that has no
    pip of its own; installation still targets that venv, never the kernel.
    """
    return [sys.executable, '-I', '-m', 'pip', '--python', str(executable),
            '--disable-pip-version-check']


def _check_venv(environment):
    environment = Path(environment)
    if environment.is_symlink() or environment.resolve() != environment.absolute():
        raise ValueError('environment_path_alias')
    marker = json.loads((environment / MARKER).read_text())
    config = {}
    for line in (environment / 'pyvenv.cfg').read_text().lower().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, separator, value = line.partition('=')
        key = key.strip()
        if not separator or key in config:
            raise ValueError('Use o ambiente isolado do projeto, sem pacotes do sistema.')
        config[key] = value.strip()
    if marker != {'backend': BACKEND} or config.get('include-system-site-packages') != 'false':
        raise ValueError('Use o ambiente isolado do projeto, sem pacotes do sistema.')
    return runtime_python(environment)


def environment_status(environment):
    if not Path(environment).exists():
        return {}
    executable = _check_venv(environment)
    code = ('import importlib.metadata as m,json,sys; '
            'print(json.dumps({"prefix":sys.prefix,"base_prefix":sys.base_prefix,'
            '"packages":{d.metadata["Name"].lower().replace("_","-"):d.version '
            'for d in m.distributions() if d.metadata["Name"]}}))')
    result = subprocess.run([str(executable), '-I', '-c', code], check=True,
                            capture_output=True, text=True, timeout=30)
    state = json.loads(result.stdout)
    if Path(state['prefix']).resolve() != Path(environment).resolve() or state['prefix'] == state['base_prefix']:
        raise ValueError('environment_not_isolated')
    return state['packages']


def install_environment(environment, repo=ROOT):
    """Explicit installation; refuse to repurpose another environment."""
    check_platform()
    locked_plan(repo)
    environment = Path(environment)
    if not environment.exists():
        if environment.is_symlink() or environment.resolve() != environment.absolute():
            raise ValueError('environment_path_alias')
        subprocess.run([sys.executable, '-I', '-m', 'venv', '--without-pip', str(environment)],
                       check=True, timeout=120)
        (environment / MARKER).write_text(json.dumps({'backend': BACKEND}))
    executable = _check_venv(environment)
    subprocess.run(_pip_command(executable) + ['install', '--require-hashes',
                   '--only-binary=:all:', '--no-index', '-r',
                   str(Path(repo) / 'requirements-colab-lock.txt')], check=True, timeout=1800)


def validate_environment(environment, repo=ROOT, *, check_cuda=True):
    check_platform()
    lock, digest = locked_plan(repo)
    status = environment_status(environment)
    wrong = [i['name'] for i in lock['distributions']
             if status.get(i['name'].lower().replace('_', '-')) != i['version']]
    if wrong:
        raise ValueError('Instale o ambiente isolado antes de continuar: ' + ', '.join(wrong))
    executable = _check_venv(environment)
    check = subprocess.run(_pip_command(executable) + ['check'],
                           capture_output=True, text=True, timeout=60)
    if check.returncode:
        raise ValueError('Conflitos no ambiente isolado: ' + check.stdout + check.stderr)
    summary = {'training_backend': BACKEND, 'lock_sha256': digest, 'packages': status}
    if check_cuda:
        code = ('import sys,json; sys.path.insert(0,sys.argv[1]); '
                'from dependencies import require_packages; require_packages(); '
                'from train import cuda_runtime; r=cuda_runtime(); import torch; '
                't=torch.ones((16,16),device=r["device"],dtype=r["dtype"]); '
                '(t@t).sum().item(); torch.cuda.synchronize(); r.pop("dtype"); '
                'print(json.dumps(r))')
        try:
            result = subprocess.run([str(executable), '-I', '-B', '-c', code,
                                     str(Path(repo) / 'ft05')], check=True,
                                    capture_output=True, text=True, timeout=120)
        except subprocess.CalledProcessError as exc:
            # This fixed probe loads no model, input text, Drive file or token.
            diagnostic = (exc.stderr or '').strip()[-2000:]
            raise ValueError('Não foi possível verificar a GPU. Nenhum treino foi iniciado.\n'
                             + diagnostic) from None
        summary['gpu'] = json.loads(result.stdout)
    return summary


def save_environment(root, summary):
    destination = Path(root) / 'environment-colab.json'
    destination.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return destination
