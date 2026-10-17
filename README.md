# Radar Classificador Colab

Crie e teste **seu próprio adaptador de IA** para classificar mensagens, usando a GPU do Google Colab e salvando os resultados no seu Google Drive. No computador, você usa o navegador.

**Estado: projeto educacional experimental.** O fluxo separado neste repositório ainda precisa concluir uma execução inteira em uma segunda conta Google e pasta nova. Não está homologado para atendimento real.

Existe também uma [pendência de dependência](SECURITY.md#aviso-de-dependência-conhecido):
a biblioteca Datasets fixada tem um aviso em carregadores de pastas, que este
exercício não usa. Uma combinação corrigida e compatível precisa ser homologada
antes de anunciar a versão estável.

## Comece por aqui

**Prévia para homologação:** [abrir Comece aqui no Colab](https://colab.research.google.com/github/ofelipepeixoto/radar-classificador-colab/blob/feat/community-initial/COMECE-AQUI.ipynb) · [testar adaptador salvo](https://colab.research.google.com/github/ofelipepeixoto/radar-classificador-colab/blob/feat/community-initial/TESTAR-ADAPTADOR.ipynb).
Esses links apontam à branch de rascunho com conteúdo verificado pelo notebook, ainda sem tag estável.

1. Leia o [passo a passo](docs/COMECE-AQUI.md).
2. Abra [COMECE-AQUI.ipynb](COMECE-AQUI.ipynb) no Colab e salve uma cópia na sua conta.
3. Siga uma célula por vez para preparar o exercício, instalar as dependências e treinar o seu adaptador.
4. Use a interface de texto e botão para testar mensagens fictícias. Em outra sessão, retome com [TESTAR-ADAPTADOR.ipynb](TESTAR-ADAPTADOR.ipynb).

Cada pessoa usa uma pasta nova no próprio Drive. O repositório não distribui o adaptador do autor, não requer sua GPU local e não abre um servidor público para a interface.

## O que a IA faz

Ela recebe uma mensagem e devolve dois campos:

```json
{"categoria":"entrega","urgencia":"media"}
```

| Campo | Valores do exercício |
| --- | --- |
| `categoria` | `entrega`, `pagamento`, `cancelamento`, `outro` |
| `urgencia` | `baixa`, `media`, `alta` |

É um classificador. Conversar com clientes, redigir respostas e executar cancelamentos ou pagamentos ficam fora deste exercício. As categorias e o modelo base são fixos nesta versão.

## O que você precisa

- Conta Google, acesso ao Colab e espaço no Drive.
- Runtime com GPU compatível. Disponibilidade, tempo e consumo de unidades dependem da sua conta e sessão; o projeto não garante execução gratuita.
- Mensagens fictícias para testar a interface.

O modelo base é `Qwen/Qwen2.5-1.5B-Instruct`; o treinamento cria um adaptador LoRA. As versões de referência vêm de uma execução anterior em Python 3.12 e GPU NVIDIA RTX PRO 6000 Blackwell. Isso não comprova compatibilidade com toda GPU ou runtime do Colab.

## Dados e limites

O corpus contém **208 exemplos sintéticos**: 162 de treino, 12 de validação e 34 de teste. Eles passaram por checagens automáticas e continuam com **`reviewed: false`**. O tutorial não pede que você preencha manualmente 216 linhas para executar esse piloto.

O piloto original obteve 32/34 acertos no teste sintético congelado e 20/24 em um ensaio exploratório posterior. São conjuntos pequenos, sintéticos e diferentes entre si. Também foram observadas urgências excessivas em pedidos sem prazo iminente. Esses números não representam desempenho com clientes reais nem garantem o mesmo resultado em outra execução.

O teste congelado serve para avaliação. Não ajuste os rótulos ou retreine com seus erros para aumentar a pontuação. Para outro domínio ou uso real, será necessário definir um protocolo novo, com dados autorizados e rótulos revisados de forma independente.

## Onde encontrar

| Conteúdo | Caminho |
| --- | --- |
| Primeiro treinamento no seu Drive | [COMECE-AQUI.ipynb](COMECE-AQUI.ipynb) |
| Reabrir o adaptador criado neste projeto | [TESTAR-ADAPTADOR.ipynb](TESTAR-ADAPTADOR.ipynb) |
| Instruções e solução de problemas | [docs/COMECE-AQUI.md](docs/COMECE-AQUI.md) |
| Plano e homologação pendente | [docs/PLANO.md](docs/PLANO.md) |
| Código, corpus e testes do FT05 | [ft05/](ft05/) |
| Contribuições | [docs/COMO-CONTRIBUIR.md](docs/COMO-CONTRIBUIR.md) |
| Segurança | [SECURITY.md](SECURITY.md) |
| Auditoria e evidências | [docs/AUDITORIA.md](docs/AUDITORIA.md) |

## Para contribuir com código

Os testes offline usam apenas a biblioteca padrão do Python; não treinam nem instalam bibliotecas de GPU:

```bash
PYTHONPATH=ft05 python -m unittest discover -s ft05/tests -v
```

No PowerShell, use `$env:PYTHONPATH = "ft05"` antes de `python -m unittest discover -s ft05/tests -v`. Quem só deseja fazer o exercício no Colab não precisa abrir um terminal.

Este projeto foi separado do FT05 de [`radar-model-router`](https://github.com/ofelipepeixoto/radar-model-router). O projeto de origem foi preservado. Consulte a licença e o registro de origem incluídos aqui antes de redistribuir.
