"""Small offline helpers for the community Colab workflow."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

from dependencies import VERSIONS, package_status, require_packages



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


def train_project(selected, runtime_python):
    """One explicit synthetic run; never overwrite a previous config or run."""
    from ft05 import load_frozen
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
    source = Path(__file__).resolve().parent
    command = ('import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); '
               'runpy.run_module(\"train\",run_name=\"__main__\")')
    subprocess.run([str(runtime_python), '-I', '-B', '-c', command, str(source),
                    str(selected['corpus']), '--config', str(config_path),
                    '--output', str(run_dir), '--frozen', str(selected['frozen']), '--train'],
                   check=True)
    result = json.loads((run_dir / 'run.json').read_text())
    if result.get('trained') is not True:
        raise ValueError('Treino incompleto; preserve a pasta para investigar.')
    return result
