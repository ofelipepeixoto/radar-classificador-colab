"""Generate clean community notebooks and a pinned source manifest. Offline."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = 'https://github.com/ofelipepeixoto/radar-classificador-colab.git'
REPO_REF = 'fix/safe-training-dependencies'  # Draft distribution; verified by content below.


def source_manifest():
    files = [path for folder in ('ft05', 'scripts')
             for path in sorted((ROOT / folder).rglob('*'))
             if path.is_file() and '__pycache__' not in path.parts
             and path.suffix in {'.py', '.json', '.jsonl', '.csv'}]
    files += [ROOT / 'requirements-colab.txt', ROOT / 'requirements-colab-lock.txt']
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def cell(kind, source):
    item = {'cell_type': kind, 'id': hashlib.sha256((kind + source).encode()).hexdigest()[:12],
            'metadata': {}, 'source': source.splitlines(True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item


def build(inference_only, manifest_sha):
    cells = []
    def md(text): cells.append(cell('markdown', text))
    def code(text): cells.append(cell('code', text))

    name = 'TESTAR-ADAPTADOR.ipynb' if inference_only else 'COMECE-AQUI.ipynb'
    md('''# Radar Classificador Colab — ''' + ('teste seu adaptador salvo' if inference_only else 'comece aqui') + '''

**Versão em preparação: homologação completa em segunda conta pendente.**
Seu PC abre o navegador; treino e inferência usam a GPU do Colab.
Este exemplo classifica mensagens em categoria e urgência. Não é um chatbot.
Use somente mensagens fictícias. Nenhum peso ou Drive privado do autor é necessário.
O corpus é sintético, sem aprovação humana; não valida atendimento real.

Salve uma cópia no seu Drive. Execute uma célula por vez, na ordem.
Cada etapa tem sua própria caixa de ativação; todas começam desmarcadas.
Não use “Executar tudo”. Leia o [guia do projeto](https://github.com/ofelipepeixoto/radar-classificador-colab/blob/fix/safe-training-dependencies/docs/COMECE-AQUI.md).
''')
    md('''## Configuração — ''' + ('informe a mesma pasta e execução do seu treino' if inference_only else 'escolha uma pasta nova') + '''

Altere somente os dois campos abaixo. Use letras, números, `_` ou `-` e `/` entre pastas.
''' + ('Este notebook aceita apenas adaptadores com selo de integridade criado pelo novo projeto. O piloto antigo do autor continua no notebook original.' if inference_only else 'Escolha um destino exclusivo. Arquivos de execuções anteriores serão preservados.'))
    code('''from pathlib import Path
import hashlib, importlib.metadata, json, re, shutil, subprocess, sys

PASTA_PROJETO = 'RadarClassificador/homologacao-001' # @param {type:"string"}
NOME_EXECUCAO = 'run-001' # @param {type:"string"}

REPO_DIR = Path('/content/radar-classificador-colab')
ENV_DIR = Path('/content/radar-classificador-env-hf')
ENV_PYTHON = ENV_DIR / 'bin/python'
if (not PASTA_PROJETO or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', p)
                           for p in PASTA_PROJETO.split('/'))
        or len(PASTA_PROJETO) > 150
        or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', NOME_EXECUCAO)):
    raise ValueError('Use nomes simples e pasta relativa como RadarClassificador/homologacao-001.')
ROOT = Path('/content/drive/MyDrive') / PASTA_PROJETO
CORPUS = ROOT / 'synthetic-machine-checked.jsonl'
FROZEN = ROOT / 'frozen'
RUN_DIR = ROOT / NOME_EXECUCAO
print('Sua pasta:', ROOT)
print('Sua execução:', RUN_DIR.name)
''')
    md('''## 1. Confira o ambiente

Selecione um runtime com GPU na interface do Colab e confira as unidades da sua conta.
Não é necessário escolher a GPU mais cara. A GPU e o tempo disponíveis variam.
Use Python 3.12 e uma GPU de classe 16 GB ou maior. O código verificará CUDA 12.9.
A nova rota usa Transformers + PEFT + TRL, sem Unsloth; a homologação GPU ainda está pendente.''')
    code('''print('Python:', sys.version.split()[0])
gpu = shutil.which('nvidia-smi')
if gpu:
    result = subprocess.run([gpu, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=30)
    print('GPU:', result.stdout.strip() if result.returncode == 0 else result.stderr.strip())
else:
    print('GPU não detectada. Selecione GPU antes de treinar ou abrir a interface.')
''')
    md('''## 2. Obtenha o código e execute os testes

Marque **OBTER_CODIGO** e execute esta célula. O conteúdo Python e os dados
são conferidos por SHA-256 antes de qualquer importação do projeto.
Uma cópia existente é validada; não há atualização ou sobrescrita automática.''')
    code(f'''OBTER_CODIGO = False # @param {{type:"boolean"}}
if OBTER_CODIGO:
    if not REPO_DIR.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', '{REPO_REF}',
                        '{REPO_URL}', str(REPO_DIR)], check=True, timeout=180)
    if REPO_DIR.is_symlink():
        raise ValueError('Pasta do código não pode ser um link.')
    # Only source files belong to this package. Reject stale or unlisted binaries.
    if any(p.suffix in {{'.pyc', '.pyo', '.so', '.pyd'}}
           for folder in ('ft05', 'scripts') for p in (REPO_DIR / folder).rglob('*') if p.is_file()):
        raise ValueError('Código compilado inesperado. Use uma sessão limpa.')
    sys.dont_write_bytecode = True
    manifest_file = REPO_DIR / 'SOURCE-MANIFEST.json'
    if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != '{manifest_sha}':
        raise ValueError('Versão do código diferente. Abra uma cópia atual do notebook em sessão limpa.')
    manifest = json.loads(manifest_file.read_text())
    for relative, expected in manifest.items():
        path = REPO_DIR / relative
        if (not path.resolve().is_relative_to(REPO_DIR.resolve()) or path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError('Arquivo de código/dados alterado: ' + relative)
    # Reject extra importable modules; a checked file must not import an unlisted one.
    actual = {{str(p.relative_to(REPO_DIR)) for folder in ('ft05', 'scripts')
               for p in (REPO_DIR / folder).rglob('*')
               if p.is_file() and '__pycache__' not in p.parts
               and p.suffix in {{'.py', '.json', '.jsonl', '.csv'}}}}
    actual.update({{'requirements-colab.txt', 'requirements-colab-lock.txt'}})
    if actual != set(manifest):
        raise ValueError('Arquivos inesperados no código. Use uma sessão limpa.')
    sys.path.insert(0, str(REPO_DIR / 'ft05'))
    commit = subprocess.check_output(['git', '-C', str(REPO_DIR), 'rev-parse', 'HEAD'],
                                     text=True, timeout=30).strip()
    print('Commit do código:', commit)
    subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                   cwd=REPO_DIR / 'ft05', check=True, timeout=120)
    print('Código e testes: OK. Nenhuma biblioteca GPU foi instalada.')
else:
    print('Marque OBTER_CODIGO e execute esta célula para continuar.')
''')
    md('''## 3. Monte o seu Drive''' + ('' if inference_only else ' e prepare o corpus') + '''

Marque **MONTAR_DRIVE**''' + ('.' if inference_only else ' e **PREPARAR_CORPUS**.') + '''
Autorize somente sua conta. O código passa a ter acesso ao Drive montado.
''' + ('A verificação inicial não carrega pesos e recusa arquivos alterados.' if inference_only else 'Serão 208 mensagens fictícias: 162 treino, 12 validação e 34 teste. Revisão humana continua falsa. A repetição valida arquivos existentes.'))
    source = '''MONTAR_DRIVE = False # @param {type:"boolean"}
'''
    if not inference_only:
        source += 'PREPARAR_CORPUS = False # @param {type:"boolean"}\n'
    source += '''if MONTAR_DRIVE:
    if not OBTER_CODIGO:
        raise ValueError('Conclua primeiro a etapa 2 nesta sessão.')
    from google.colab import drive
    drive.mount('/content/drive')
    from community import paths
    selected = paths(PASTA_PROJETO, NOME_EXECUCAO)
    ROOT, CORPUS, FROZEN, RUN_DIR = [selected[k] for k in ('root', 'corpus', 'frozen', 'run')]
'''
    if inference_only:
        source += '''    from personal_smoke import preflight
    checked = preflight(RUN_DIR, CORPUS, FROZEN, versions=False)
    print('Corpus, split e selo do adaptador conferidos. Modelo:', checked['meta']['model'])
'''
    else:
        source += '''    if PREPARAR_CORPUS:
        from community import prepare_project
        summary = prepare_project(ROOT)
        print('Corpus e partições:', summary)
    else:
        print('Marque também PREPARAR_CORPUS para preparar ou verificar os arquivos.')
'''
    source += '''else:
    print('Drive não montado por esta célula. Marque MONTAR_DRIVE para continuar.')
'''
    code(source)
    md('''## 4. Confira o plano das dependências

Marque **CONFERIR_DEPENDENCIAS** e **REVISAR_PLANO**.
A preparação desta versão resolveu as 94 dependências. Confira o plano fixado por hashes;
essa etapa não instala nem baixa os pacotes CUDA. O ambiente fica separado do Python do Colab.
Datasets 5.0.1 substitui a versão afetada. Veja o risco residual documentado de Accelerate em `SECURITY.md`.''')
    code('''CONFERIR_DEPENDENCIAS = False # @param {type:"boolean"}
REVISAR_PLANO = False # @param {type:"boolean"}
PLANO_REVISADO = False
if CONFERIR_DEPENDENCIAS:
    if not MONTAR_DRIVE or not FROZEN.is_dir():
        raise ValueError('Conclua a etapa 3 antes de continuar.')
    from colab_environment import check_platform, locked_plan, environment_status
    check_platform()
    lock, lock_sha = locked_plan(REPO_DIR)
    present = environment_status(ENV_DIR)
    if REVISAR_PLANO:
        for item in lock['distributions']:
            name = item['name']
            print(name, 'necessário:', item['version'], '| presente:', present.get(name.lower().replace('_', '-'), 'ausente'))
        print('Plano:', len(lock['distributions']), 'pacotes | SHA-256:', lock_sha)
        print('Nada foi instalado. O ambiente padrão do Colab será preservado.')
        PLANO_REVISADO = True
    else:
        print('Marque também REVISAR_PLANO para conferir as versões fixadas.')
else:
    print('Marque CONFERIR_DEPENDENCIAS para verificar o ambiente.')
''')
    md('''## 5. Instale no ambiente isolado e verifique CUDA

Na primeira execução, marque **INSTALAR_DEPENDENCIAS**. O download pode levar vários minutos.
Todos os pacotes e hashes vêm do plano anterior, sem alterar o Python do notebook.
Nas repetições, deixe a caixa desmarcada para verificar o ambiente já instalado.
A etapa só libera o treino após conferir versões, conflitos e GPU. Não requer reiniciar o kernel.''')
    code('''INSTALAR_DEPENDENCIAS = False # @param {type:"boolean"}
AMBIENTE_OK = False
if CONFERIR_DEPENDENCIAS and PLANO_REVISADO:
    from colab_environment import install_environment, validate_environment, save_environment
    if INSTALAR_DEPENDENCIAS:
        install_environment(ENV_DIR, REPO_DIR)
    environment = validate_environment(ENV_DIR, REPO_DIR)
    print('GPU e precisão:', environment['gpu'])
    print('Dependências e pip check: OK. Registro privado:', save_environment(ROOT, environment))
    AMBIENTE_OK = True
    print('Treinamento: não iniciado. Ambiente isolado pronto.')
else:
    print('Conclua e revise o plano da etapa 4 primeiro.')
''')
    if not inference_only:
        md('''## 6. Treine o seu adaptador — ação explícita, com GPU

Confira saldo e versões. Marque as três caixas abaixo e execute somente esta célula.
Um treino concluído não deve ser repetido. A pasta de execução não pode já existir.
O treino compara base e adaptado no mesmo teste sintético e salva artefatos no seu Drive.''')
        code('''TREINAR = False # @param {type:"boolean"}
ORCAMENTO_GPU_CONFIRMADO = False # @param {type:"boolean"}
DEPENDENCIAS_REVISADAS = False # @param {type:"boolean"}
if TREINAR:
    if not (MONTAR_DRIVE and PREPARAR_CORPUS and globals().get('AMBIENTE_OK', False)
            and ORCAMENTO_GPU_CONFIRMADO and DEPENDENCIAS_REVISADAS):
        raise ValueError('Conclua as etapas anteriores e confirme orçamento e dependências.')
    from community import train_project
    result = train_project(selected, ENV_PYTHON)
    print('Treino concluído:', result['trained'])
    print('Adaptador no seu Drive:', RUN_DIR / 'adapter')
    print('Manifesto:', RUN_DIR / 'run.json')
else:
    print('Treino desligado. Nada foi treinado.')
''')
    md('''## ''' + ('6' if inference_only else '7') + '''. Abra a interface e teste

Marque **ABRIR_INTERFACE**. O modelo é carregado uma vez por sessão.
Digite uma mensagem fictícia e clique em **Classificar**. Você verá categoria, urgência e tempo.
A interface não cria servidor nem link público e não grava histórico no Drive.
Saídas do notebook podem ser salvas pelo Colab: limpe-as antes de compartilhar sua cópia.''')
    code('''ABRIR_INTERFACE = False # @param {type:"boolean"}
if ABRIR_INTERFACE:
    if not MONTAR_DRIVE or not globals().get('AMBIENTE_OK', False):
        raise ValueError('Conclua as etapas de código, Drive e dependências primeiro.')
    requested_identity = (str(RUN_DIR.resolve()), hashlib.sha256((RUN_DIR / 'run.json').read_bytes()).hexdigest())
    if 'panel' in globals() and globals().get('panel_identity') != requested_identity:
        raise ValueError('Outra execução já está carregada. Reinicie a sessão para trocar de adaptador.')
    if 'panel' in globals() and getattr(getattr(panel, '_ft05_worker', None), 'closed', True):
        panel.close()
        del panel
    if 'panel' not in globals():
        from colab_ui import launch
        panel = launch(ENV_PYTHON, RUN_DIR, CORPUS, FROZEN)
        panel_identity = requested_identity
    else:
        print('Interface já carregada nesta sessão. Use o campo de mensagem acima.')
else:
    print('Interface desligada. Marque ABRIR_INTERFACE para testar seu adaptador.')
''')
    md('''## Ao terminar

Confira no Drive `run.json`, `adapter/` e `comparison.json`. Não compartilhe a pasta do Drive.
Para reabrir o adaptador salvo sem treinar novamente, use `TESTAR-ADAPTADOR.ipynb` com os mesmos nomes de pasta e execução.
Encerre o runtime pelo menu do Colab para liberar a GPU; salvar o notebook não salva a VM.
Uma instalação nova pode ser necessária na próxima sessão. Não retreine com os erros do teste congelado.

O modelo pode superpriorizar cancelamentos sem prazo iminente. A qualidade real depende de
dados permitidos e rótulos independentes; veja `docs/INDEPENDENT-EVAL.md`.
''')
    return {'cells': cells, 'metadata': {'colab': {'name': name},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}


def generate():
    content = json.dumps(source_manifest(), ensure_ascii=False, indent=2) + '\n'
    (ROOT / 'SOURCE-MANIFEST.json').write_text(content, encoding='utf-8')
    digest = hashlib.sha256(content.encode()).hexdigest()
    for inference_only, name in ((False, 'COMECE-AQUI.ipynb'), (True, 'TESTAR-ADAPTADOR.ipynb')):
        (ROOT / name).write_text(json.dumps(build(inference_only, digest), ensure_ascii=False, indent=1) + '\n',
                                 encoding='utf-8')
    print('Notebooks limpos e manifesto gerados:', digest)


if __name__ == '__main__':
    generate()
