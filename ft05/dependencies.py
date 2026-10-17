"""Pinned environment for the HF-only training route. No GPU imports."""
import importlib.metadata

BACKEND = 'transformers-peft-lora-v1'
TORCH_INDEX = 'https://download.pytorch.org/whl/cu129'
VERSIONS = {
    'torch': '2.13.0+cu129',
    'transformers': '5.13.1',
    'trl': '1.15.0',
    'datasets': '5.0.1',
    'peft': '0.19.1',
    'accelerate': '1.15.0',
    'ipywidgets': '8.1.8',
    'setuptools': '83.0.0',
}


def package_status():
    current = {}
    for package in VERSIONS:
        try:
            current[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            current[package] = 'ausente'
    return current


def require_packages():
    current = package_status()
    wrong = [name for name in VERSIONS if current[name] != VERSIONS[name]]
    if wrong:
        raise ValueError('Dependências incompatíveis: ' + ', '.join(wrong))
