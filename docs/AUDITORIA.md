# Auditoria da extração FT05 — 17/10/2026

## Decisão

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

### Pendência DEP-01

O aviso descreve leitura fora de diretório por metadado `file_name` em builders
de pastas. O feed informa correção em Datasets `5.0.1`. A metadata oficial no
PyPI de Unsloth `2026.10.6` (versão mais recente consultada) exige
`datasets!=4.0.*,!=4.1.0,!=4.4.*,!=4.5.0,<5.0.0,>=3.4.1`.
Forçar `5.0.1` contradiz essa exigência. A tentativa de atualização foi descartada
da configuração; não foi instalado pacote GPU ou alterado o ambiente do autor.
Não há exploração observada neste projeto. Não expandir o fluxo a datasets externos.
Antes de versão estável, resolver e homologar uma combinação corrigida, sem
ignorar o resolver ou o pip check. O novo repositório não anuncia zero vulnerabilidades.

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

O lock transitivo real deve ser gerado e auditado na sessão Colab homologada.
Um pin de versão não prova ausência de vulnerabilidade. O aviso de geração sobre
max_new_tokens/max_length não foi ocultado ou alterado sem teste da biblioteca GPU.

## Revisão e próximo passo

Revisar a PR de extração. A `main` foi protegida: PR e checks `checks (3.11)`/`checks (3.12)` obrigatórios, base atualizada, conversas resolvidas, sem force push ou exclusão, incluindo administradores. A regra não exige uma aprovação GitHub de terceiro (contagem 0, repositório pessoal); revisão humana continua uma etapa operacional antes de integrar.
Resolver DEP-01 e executar o [roteiro de homologação](PLANO.md) em outra conta
autorizada/pasta nova, incluindo retomada em runtime novo. Só então fixar tag e
links estáveis para divulgar o tutorial. O `run-002-g4` e o Drive original ficam intactos.

## Evidências remotas conferidas

- [PR #1 em rascunho](https://github.com/ofelipepeixoto/radar-classificador-colab/pull/1), sem merge ou release.
- Código publicado em `d5f473e76d4042f2282817e24ee0845715b80d66`; sincronização da página inicial em `9bbea77968941f4318051f55935e0af56fe3adcf`, sem alteração da árvore de código.
- [CI push aprovado](https://github.com/ofelipepeixoto/radar-classificador-colab/actions/runs/39061633205) e [CI PR aprovado](https://github.com/ofelipepeixoto/radar-classificador-colab/actions/runs/39061639051), matriz Python 3.11/3.12.
- Clone independente do GitHub: a própria célula 2 do notebook validou o manifesto e executou os 51 testes com sucesso. Publication checks passaram em 58 arquivos de texto; nenhuma biblioteca GPU foi instalada.
- Metadata do repositório confirmou visibilidade pública, scanner de segredos e proteção de push de segredos ativos. Proteção de main foi relida após configurar. Não foi feito teste destrutivo de force push ou publicação de segredo real.
- As duas refs da origem foram relidas e continuavam nos mesmos SHAs registrados, sem alteração pela extração. O Drive não foi acessado.
- Pendências rastreadas: [dependências corrigidas, issue #3](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/3) e [homologação Colab, issue #2](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/2).
