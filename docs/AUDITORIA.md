# Auditoria FT05 — extração e nova rota de dependências

## Atualização: Transformers + PEFT + TRL

A branch `fix/safe-training-dependencies` prepara o tratamento da
[issue #3](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/3)
em uma PR sobre a PR #1 de extração. A mudança remove Unsloth da rota de
treino e inferência, atualiza Datasets para `5.0.1` e usa LoRA em 16 bits,
sem quantização de 4/8 bits. A base Qwen na revisão fixa e o corpus sintético
original não foram alterados. Os resultados históricos de 32/34 e 20/24 não
são resultados do novo backend; seu desempenho ainda precisa ser medido.

### Instalação e fronteiras

- Ambiente dedicado `/content/radar-classificador-env-hf`, sem
  `system-site-packages`. Os pacotes do kernel Colab não são substituídos.
- Criação com `venv --without-pip`: o pip do kernel gerencia o Python do
  ambiente por `--python`, conforme a [documentação oficial do pip](https://pip.pypa.io/en/stable/topics/python-option/).
  Essa correção contorna a ausência observada de `ensurepip`, sem instalar
  pip dentro do ambiente nem acrescentar dependências ao lock.
- `requirements-colab-lock.txt`: 94 distribuições com URLs diretas e hashes,
  específicas para Python 3.12/Linux x86_64. O instalador exige
  `--require-hashes` e `--no-index`; `ft05/environment-lock.json` registra a
  composição esperada.
- O plano de instalação já foi resolvido na preparação. Sua revisão mostra
  versões e hashes sem baixar a distribuição CUDA. Instalação, `pip check`,
  conferência das versões e verificação CUDA precedem `AMBIENTE_OK`.
- Treino e inferência são executados pelo Python do ambiente separado. A
  interface de widgets permanece no notebook, sem servidor ou túnel público.
- Receita para GPU de classe 16 GB ou maior; a verificação usa piso de 14 GiB
  utilizáveis (15.032.385.536 bytes), permitindo a reserva de memória do driver.
  Na T4 observada, Torch informou 15.637.086.208 bytes (~14,56 GiB), apesar dos
  15360 MiB no nvidia-smi. Esse piso não garante memória de pico suficiente.
  Uma operação FP16 com a distribuição CUDA 12.9 foi executada com sucesso
  no driver 580.82.07; o treino completo ainda precisa terminar.

O ambiente Python separado evita herdar conflitos de pacotes; não é um
sandbox de segurança e mantém o acesso da sessão ao Drive montado.

### Dependências e risco residual

| Item | Tratamento e limite |
| --- | --- |
| Datasets `4.8.5`, DEP-01 histórico | Substituído por `5.0.1`, a versão corrigida informada no advisory. Remover Unsloth elimina a restrição que impedia essa combinação. O fluxo continua limitado a `Dataset.from_dict` com texto sintético validado |
| Accelerate `1.15.0` | Risco residual para `weight_map` de checkpoints fragmentados. O feed GHSA lista até `1.14.0` e nenhuma versão corrigida; a inspeção do wheel `1.15.0` ainda encontrou o caminho de código relevante. Ausência de alerta OSV nessa versão não prova correção |
| Carregamento de pesos | Somente base conhecida na revisão fixa e adaptador local selado com dois arquivos de nomes fixos. Índices externos de fragmentos, pickle e dispatch de checkpoints externos são recusados ou excluídos da rota |

Esses controles limitam a entrada do exercício; **a biblioteca Accelerate não
é declarada corrigida**. Não houve conclusão de ausência de vulnerabilidades.
Consulte [SECURITY.md](../SECURITY.md#risco-residual-em-accelerate) e o
[advisory Accelerate](https://github.com/advisories/GHSA-4j2p-28q2-5m79).

### Evidências desta atualização

| Verificação | Evidência disponível |
| --- | --- |
| Resolução do ambiente | 94 distribuições resolvidas e hashes registrados para Python 3.12/Linux x86_64, sem GPU |
| Testes offline após os ajustes de homologação | 93 testes: 92 aprovados e 1 integração HF opcional omitida; a integração real anterior está registrada separadamente abaixo. Dois notebooks validados com nbformat |
| Auditoria transitiva | [Consulta OSV e limites](dependency-audit-2026-10-17.json): 94 distribuições sem correspondência no feed; Torch normalizado para 2.13.0. Não analisa binários CUDA; inspeção manual encontrou o residual Accelerate |
| Análise estática | Bandit 1.9.4: 20 observações baixas, nenhuma média/alta. Chamadas subprocesso usam listas, sem shell; caminhos de script são locais conferidos. Uso de random é para separar exemplos, sem função criptográfica |
| Integração real Hugging Face em CPU | 1 teste aprovado em 3,649 s: Qwen minúsculo aleatório e local, backward + AdamW alteraram parâmetros LoRA; `SFTTrainer` + Datasets `5.0.1` prepararam `input_ids`; safetensors gravado, selado e reaberto com logits iguais; geração real sem conflito `max_length`/`max_new_tokens`. `socket.connect` bloqueado e não chamado |
| Ambiente Colab observado | Python `3.12.13`, T4 com `15.360 MiB`, driver `580.82.07`, pip do kernel `24.1.2`. Montagem do Drive e preparação dos 208 exemplos concluídas, com hashes originais preservados. Instalação completa e treino ainda não confirmados |
| Homologação Colab | Pendente na [issue #2](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/2): concluir instalação e validação CUDA, treinar com GPU, salvar, encerrar a sessão e reabrir o adaptador em outra sessão |
| Qualidade do novo adaptador | Ainda não medida. Não reutilizar a pontuação histórica como evidência desta rota |

### Ajustes encontrados durante a homologação

| Falha ou limite | Tratamento preparado | Validação restante |
| --- | --- | --- |
| Colab retornou `No module named ensurepip` ao criar venv com pip | Criar venv sem pip e usar o pip do kernel com destino `--python` explícito; pacotes do kernel e lock de 94 distribuições preservados. Testes do helper passaram com venv real sem pip e sem rede | Confirmar instalação completa no Colab |
| Opção `save_safetensors` ausente na API de `SFTConfig` fixada | Remover a opção do construtor; manter gravação explícita do adaptador em safetensors | Executar treinamento real com a versão fixada |
| BF16 aceito por emulação em GPU sem suporte nativo | Consultar `including_emulation=False`; T4 seleciona FP16 | Conferir saída CUDA e treinamento na T4 |
| Uma execução incompleta ou com métricas não finitas poderia parecer concluída | Verificar passos de otimizador, perdas e métricas antes de declarar o treino salvo | Exercitar treino real e conferir seus metadados |
| Processo encerrado deixava painel indisponível; bytecode poderia surgir na inferência | Permitir reabrir painel encerrado e iniciar worker com `-B` | Verificar interface após treino e em sessão nova |

Esses ajustes não significam que `SFTTrainer.train()` já rodou no Colab.
Nenhuma instalação completa ou pontuação do novo backend é afirmada a partir
da montagem do Drive e da preparação do corpus.

O teste CPU executou operações reais de Transformers e PEFT e o construtor/
preparo de dados do TRL. **Não executou `SFTTrainer.train()`**: o caminho de
treino dessa versão de TRL depende de Triton/CUDA. Portanto ele não substitui
a homologação de treinamento na GPU nem valida a base Qwen 1,5B.

Reabrir um adaptador concluído significa fazer inferência novamente. Retomar
um treinamento interrompido a partir de checkpoint não está implementado.
Preserve tentativas parciais e diagnostique antes de criar outra execução.

## Extração inicial — registro histórico de 17/10/2026

As seções abaixo registram a extração anterior e seus resultados remotos.
Suas contagens de testes, versões antigas e pendência DEP-01 não descrevem
a configuração atual: foram preservadas como histórico. O tratamento novo
está documentado acima e ainda precisa da homologação GPU.

### Decisão da extração

Extração preparada para revisão e homologação. O código do roteador, n8n e
Laya não foi copiado. Não houve alteração no repositório de origem nesta extração.
Não há release estável nem validação de atendimento real. A homologação Colab
e a pendência de dependências abaixo impedem anunciar o tutorial como pronto.

## Escopo e procedência

- Fontes fixadas no [registro de origem](../SOURCE-PROVENANCE.json): PR #8 em
  `8fbfe01ed12242d772d386ab2b763a8467f95430`, mais quatro arquivos públicos
  da PR #9 em `5e34d452cb83376e084480a1bbc04034e03902d6`.
- Código, dados fictícios, testes e documentação. Nenhum peso, token, conta
  Google, planilha revisada ou arquivo privado do Drive foi acessado ou copiado.
- Notebooks públicos regenerados a partir de `scripts/build_notebooks.py`,
  com IDs de células, outputs vazios e contagens de execução nulas.
- Fronteiras: GitHub distribui código; Colab executa Python e baixa dependências/
  base; o Drive do operador guarda corpus, split e adaptador. Widgets ficam
  dentro do notebook e não abrem servidor/túnel ou URL pública de inferência.
- A montagem permite acesso ao Drive pelos códigos da sessão. Hashes detectam
  mudanças, não autenticam autores nem protegem contra quem modifica também o selo.

## Achados e tratamento

| ID | Severidade e evidência | Correção e verificação | Estado |
| --- | --- | --- | --- |
| SEC-01 | Média: preflight antigo aceitava pesos/config alterados desde que não vazios | Novos treinos selam SHA-256 de dois nomes fixos; inferências verificam hashes, base, revisão e LORA antes de importar GPU; alteração de bytes recusada em teste | Corrigido na cópia; legado preservado no projeto original |
| SEC-02 | Baixa: interface não aplicava filtro de e-mail/segredos usado no smoke | Validador compartilhado e teste de callback com e-mail fictício; botão volta a funcionar após rejeição | Corrigido; filtro heurístico não anonimiza |
| SEC-03 | Média: clone de branch móvel não fixava código; bytecode não listado poderia contornar o hash do fonte | Manifesto SHA-256 embutido no notebook, arquivos Python/dados/scripts conferidos antes de imports; binários inesperados recusados; caches não são escritos pelo notebook/testes | Corrigido; distribuição ainda em branch de rascunho, sem assinatura |
| SEC-04 | Baixa: etapa GPU podia avançar após falha de pip check | Estado AMBIENTE_OK só é verdadeiro depois de versões e pip check aprovados; cenário de conflito impede treino em teste | Corrigido |
| SEC-05 | Baixa: alterar execução podia reutilizar painel do modelo anterior | Painel vinculado ao caminho e hash de run.json; troca exige reinício e é testada sem GPU | Corrigido |
| SEC-06 | Baixa: configuração de destinos exigia limites explícitos | Nomes relativos validados, sem .., absolutos ou escape por symlink; execução não sobrescreve pasta/config existente | Corrigido nos helpers e notebooks |
| DEP-01 | Média, confirmada no feed e fontes abaixo: datasets 4.8.5 afetado por GHSA-379c-qx7v-6h59 / CVE-2026-66007 | Caminho vulnerável não exposto: só Dataset.from_dict com texto sintético validado, sem builders de pasta/load_dataset/save_to_disk/push_to_hub. Testes impedem metadado file_name e adição dessas APIs | Mitigado no escopo; biblioteca não corrigida |

### DEP-01 na extração inicial

O aviso descreve leitura fora de diretório por metadado `file_name` em builders
de pastas. O feed informa correção em Datasets `5.0.1`. A metadata oficial no
PyPI de Unsloth `2026.10.6` (versão mais recente consultada) exige
`datasets!=4.0.*,!=4.1.0,!=4.4.*,!=4.5.0,<5.0.0,>=3.4.1`.
Forçar `5.0.1` contradiz essa exigência. A tentativa de atualização foi descartada
da configuração; não foi instalado pacote GPU ou alterado o ambiente do autor.
Não há exploração observada neste projeto. Não expandir o fluxo a datasets externos.
Naquela etapa, a combinação corrigida ainda não havia sido preparada. A nova
rota descrita acima trata essa restrição sem forçar a instalação de pacotes
incompatíveis. O repositório não anuncia zero vulnerabilidades.

Fontes consultadas:

- [Advisory GitHub](https://github.com/advisories/GHSA-379c-qx7v-6h59), confirmado também pela API pública GitHub.
- [Correção no repositório oficial Datasets](https://github.com/huggingface/datasets/pull/8325).
- [Feed OSV/PyPI](https://osv.dev/vulnerability/PYSEC-2026-3716).
- [Metadata Unsloth 2026.10.6](https://pypi.org/pypi/unsloth/2026.10.6/json).

## Verificações executadas

Ambiente local de preparação: Python 3.12.14, sem GPU. Ferramentas de auditoria
instaladas em diretórios temporários, fora do projeto; nenhuma biblioteca GPU instalada.

| Verificação | Evidência |
| --- | --- |
| Suite offline | `PYTHONPATH=ft05 python -B -m unittest discover -s ft05/tests -v`; 51 testes aprovados localmente; resultado remoto registrado na PR |
| Integridade/negativos | Alteração de corpus, split, config e pesos; legado sem selo; JSON inválido; destinos que escapam; dependências ausentes/conflitantes; troca silenciosa de painel |
| Treino/inferência em doubles CPU | Fluxo do treino salva manifesto e selo; callback da interface valida e recupera botão. Não é teste de modelo/GPU |
| Notebooks | nbformat 5.11.1 validou schema dos dois arquivos; todas as células Python compiladas e execução padrão local sem rede/Drive/treino |
| Repetição de preparação | 208 mensagens, split 162/12/34; repetição conserva bytes e hash; adulteração do teste é recusada |
| Bandit 1.9.4 | 1.711 linhas de produção/scripts analisadas no primeiro passe, 0 avisos médios/altos e 10 baixos. Triagem: subprocess sem shell com argumentos locais/constantes e random.Random usado apenas em amostragem reprodutível |
| pip-audit 2.10.1 | Consulta sem resolução: cinco dependências diretas auditáveis, um aviso em Datasets. Torch 2.11.0+cu128 não encontrado no feed PyPI e não auditado. Não cobre transitivas nem binários CUDA |
| Conteúdo público | `scripts/check_publication.py`: tripwire de segredos comuns, ausência de pesos/arquivos privados e outputs de notebooks; não é garantia de ausência de todo segredo |
| CI preparado | Python 3.11/3.12, token somente leitura, checkout sem credenciais persistidas, actions fixadas por SHA e limite de dez minutos. CI remoto aprovado em Python 3.11 e 3.12 (links abaixo) |

Na extração inicial, o lock transitivo completo ainda não existia. A nova
rota agora inclui um lock com hashes, cuja execução CUDA precisa de homologação.
Um pin de versão não prova ausência de vulnerabilidade. O aviso de geração sobre
max_new_tokens/max_length não foi ocultado ou alterado sem teste da biblioteca GPU.

## Revisão e próximo passo

Revisar a PR de extração. A `main` foi protegida: PR e checks `checks (3.11)`/`checks (3.12)` obrigatórios, base atualizada, conversas resolvidas, sem force push ou exclusão, incluindo administradores. A regra não exige uma aprovação GitHub de terceiro (contagem 0, repositório pessoal); revisão humana continua uma etapa operacional antes de integrar.
O tratamento proposto para DEP-01 está na nova rota acima. Execute o
[roteiro de homologação](PLANO.md) em outra conta autorizada/pasta nova,
incluindo reabertura do adaptador em runtime novo. Só então fixar tag e
links estáveis para divulgar o tutorial. O `run-002-g4` e o Drive original ficam intactos.

## Evidências remotas conferidas

- [PR #1 em rascunho](https://github.com/ofelipepeixoto/radar-classificador-colab/pull/1), sem merge ou release.
- Código publicado em `d5f473e76d4042f2282817e24ee0845715b80d66`; sincronização da página inicial em `9bbea77968941f4318051f55935e0af56fe3adcf`, sem alteração da árvore de código.
- [CI push aprovado](https://github.com/ofelipepeixoto/radar-classificador-colab/actions/runs/39061633205) e [CI PR aprovado](https://github.com/ofelipepeixoto/radar-classificador-colab/actions/runs/39061639051), matriz Python 3.11/3.12.
- Clone independente do GitHub: a própria célula 2 do notebook validou o manifesto e executou os 51 testes com sucesso. Publication checks passaram em 58 arquivos de texto; nenhuma biblioteca GPU foi instalada.
- Metadata do repositório confirmou visibilidade pública, scanner de segredos e proteção de push de segredos ativos. Proteção de main foi relida após configurar. Não foi feito teste destrutivo de force push ou publicação de segredo real.
- As duas refs da origem foram relidas e continuavam nos mesmos SHAs registrados, sem alteração pela extração. O Drive não foi acessado.
- Pendências rastreadas: [dependências corrigidas, issue #3](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/3) e [homologação Colab, issue #2](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/2).
