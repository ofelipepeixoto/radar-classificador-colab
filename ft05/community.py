"""Small offline helpers for the community Colab workflow."""
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

VERSIONS = {'unsloth': '2026.10.6', 'torch': '2.11.0+cu128',
            'transformers': '5.13.1', 'trl': '1.15.0',
            'datasets': '4.8.5', 'peft': '0.19.1'}


def paths(folder, run_name, drive_root=Path('/content/drive/MyDrive')):
    """Keep user-selected output inside MyDrive and reject traversal/aliases."""
    if (type(folder) is not str or not folder or len(folder) > 150
            or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', part)
                   for part in folder.split('/'))):
        raise ValueError('Use uma pasta relativa, como RadarClassificador/meu-piloto-001.')
    if type(run_name) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', run_name):
        raise ValueError('Nome da execução inválido; use run-001, por exemplo.')
    base = Path(drive_root).resolve()
    root = Path(drive_root) / folder
    if not root.resolve().is_relative_to(base) or root.resolve() == base:
        raise ValueError('O destino precisa estar dentro do seu MyDrive.')
    run = root / run_name
    if not run.resolve().is_relative_to(root.resolve()):
        raise ValueError('O destino da execução saiu da pasta do projeto.')
    return {'root': root, 'corpus': root / 'synthetic-machine-checked.jsonl',
            'frozen': root / 'frozen', 'run': run}


def prepare_project(root):
    from auto_pilot import write_auto_dataset
    from ft05 import prepare, load_frozen
    root = Path(root)
    corpus, frozen = root / 'synthetic-machine-checked.jsonl', root / 'frozen'
    root.mkdir(parents=True, exist_ok=True)
    audit = write_auto_dataset(corpus)
    if not frozen.exists():
        prepare(corpus, frozen, automated=True)
    parts, split_hash = load_frozen(corpus, frozen)
    counts = {name: len(rows) for name, rows in parts.items()}
    if (audit['included'] != 208 or counts != {'train': 162, 'validation': 12, 'test': 34}
            or any(row['reviewed'] for rows in parts.values() for row in rows)):
        raise ValueError('Corpus inesperado; preserve a pasta e investigue.')
    return {'counts': counts, 'corpus_sha256': audit['dataset_sha256'],
            'split_sha256': split_hash, 'human_reviewed': False}


def package_status():
    status = {}
    for name in VERSIONS:
        try:
            status[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            status[name] = 'ausente'
    return status


def require_packages():
    found = package_status()
    wrong = [name for name in VERSIONS if found[name] != VERSIONS[name]]
    if wrong:
        raise ValueError('Confira/instale as dependências antes de continuar: ' + ', '.join(wrong))


def save_environment(root):
    """Record a lock of installed distributions without shell URLs or env dumps."""
    packages = sorted({(dist.metadata['Name'], dist.version)
                       for dist in importlib.metadata.distributions()
                       if dist.metadata['Name']})
    destination = Path(root) / 'requirements-lock-colab.txt'
    destination.write_text(''.join(f'{name}=={version}\n' for name, version in packages),
                           encoding='utf-8')
    return destination


def train_project(selected):
    """One explicit synthetic run; never overwrite a previous config or run."""
    from ft05 import load_frozen
    from train import run
    require_packages()
    load_frozen(selected['corpus'], selected['frozen'])
    run_dir = selected['run']
    config_path = selected['root'] / (run_dir.name + '-config.json')
    if run_dir.exists() or config_path.exists():
        raise ValueError('Execução/configuração já existe. Preserve-a e escolha outro NOME_EXECUCAO.')
    config = json.loads(Path(__file__).with_name('training-config.json').read_text())
    config.update(gpu_budget_approved=True, dependency_reviewed=True,
                  package_versions=VERSIONS, gguf_export_approved=False)
    with config_path.open('x', encoding='utf-8') as file:
        json.dump(config, file, indent=2)
        file.write('\n')
    return run(selected['corpus'], config_path, run_dir, execute=True,
               frozen_dir=selected['frozen'])
