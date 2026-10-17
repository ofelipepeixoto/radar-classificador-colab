"""Local adapter integrity checks; hashes detect changes, not publisher identity.

The training run records only two fixed filenames. A manifest cannot redirect
verification to another path, and legacy adapters are never silently sealed.
"""
import hashlib
import re
from pathlib import Path

from ft05 import strict_json

ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_hashes(adapter, model, revision):
    """Validate local files and base provenance before hashing saved artifacts."""
    adapter = Path(adapter)
    for name in ADAPTER_FILES:
        path = adapter / name
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError('adapter_incomplete: faltam arquivos locais não vazios do adaptador')
    config = strict_json((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
    if (type(config) is not dict
            or config.get('base_model_name_or_path') != model
            or config.get('peft_type') != 'LORA'
            or config.get('revision') not in (None, revision)):
        raise ValueError('adapter_base_mismatch: a configuração não corresponde à base fixada')
    return {name: sha256(adapter / name) for name in ADAPTER_FILES}


def verify_adapter(adapter, expected, model, revision):
    """Reject legacy manifests or modified artifacts before GPU imports."""
    if expected is None:
        raise ValueError(
            'adapter_integrity_missing: este adaptador legado não tem hashes registrados '
            'no treino; preserve-o e use o notebook original. Este projeto exige um '
            'novo treino próprio com manifesto de integridade.')
    if (type(expected) is not dict or set(expected) != set(ADAPTER_FILES)
            or any(type(value) is not str or not re.fullmatch(r'[a-f0-9]{64}', value)
                   for value in expected.values())):
        raise ValueError('adapter_integrity_manifest_invalid')
    actual = adapter_hashes(adapter, model, revision)
    if actual != expected:
        raise ValueError('adapter_changed_after_training: config ou pesos do adaptador mudaram')
    return actual
