# Radar Classificador Colab

Crie e teste **seu próprio adaptador de IA** para classificar mensagens, usando a GPU do Google Colab e salvando os resultados no seu Google Drive. No computador, você usa o navegador.

**Estado: projeto educacional experimental.** O fluxo separado neste repositório ainda precisa concluir uma execução inteira em uma segunda conta Google e pasta nova. Não está homologado para atendimento real.

A nova rota usa **Transformers + PEFT + TRL**, com LoRA em 16 bits e ambiente
isolado. Ela remove a dependência de Unsloth e permite Datasets `5.0.1`.
Há um [risco residual em Accelerate](SECURITY.md#risco-residual-em-accelerate),
limitado pelo formato e pela origem dos pesos aceitos. A biblioteca não é
declarada corrigida. Instalação e execução na GPU ainda precisam de homologação.

## Comece por aqui

**Prévia para homologação:** [abrir Comece aqui no Colab](https://colab.research.google.com/github/ofelipepeixoto/radar-classificador-colab/blob/fix/safe-training-dependencies/COMECE-AQUI.ipynb) · [testar adaptador salvo](https://colab.research.google.com/github/ofelipepeixoto/radar-classificador-colab/blob/fix/safe-training-dependencies/TESTAR-ADAPTADOR.ipynb).
Esses links apontam à branch de rascunho com conteúdo verificado pelo notebook, ainda sem tag estável.

1. Leia o [passo a passo](docs/COMECE-AQUI.md).
2. Abra [COMECE-AQUI.ipynb](COMECE-AQUI.ipynb) no Colab e salve uma cópia na sua conta.
3. Siga uma célula por vez para preparar o exercício, instalar as dependências e treinar o seu adaptador.
4. Use a interface de texto e botão para testar mensagens fictícias. Em outra sessão, reabra o adaptador com [TESTAR-ADAPTADOR.ipynb](TESTAR-ADAPTADOR.ipynb).

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
- Python 3.12 em runtime Colab Linux x86_64, com GPU de classe 16 GB ou maior e driver compatível com a distribuição CUDA 12.9 usada pela receita. O notebook verifica a memória em bytes; GPUs de 16 GB podem aparecer com aproximadamente 15 GiB. Isso não garante que a receita caiba ou funcione em toda T4 ou outra GPU.
- Tempo e unidades de computação na sua conta. Disponibilidade e consumo variam; o projeto não garante execução gratuita.
- Mensagens fictícias para testar a interface.

O modelo base continua `Qwen/Qwen2.5-1.5B-Instruct`, na revisão fixada. O treinamento cria um adaptador LoRA sem quantização de 4/8 bits. A instalação usa um ambiente Python separado em `/content/radar-classificador-env-hf`, sem herdar os pacotes do kernel Colab. Treino e inferência rodam nesse ambiente; a interface continua no navegador.

O lock contém 94 distribuições com hashes para Python 3.12/Linux x86_64. A resolução foi verificada sem GPU. Um teste real em CPU também confirmou atualização LoRA, gravação, reabertura e geração com um modelo minúsculo local, além da preparação de dados pelo TRL. Isso não homologa o treino TRL em GPU, CUDA, memória ou qualidade do adaptador 1,5B. Reabrir um adaptador salvo é suportado; continuar um treinamento a partir de checkpoint não está implementado.

## Dados e limites

O corpus contém **208 exemplos sintéticos**: 162 de treino, 12 de validação e 34 de teste. Eles passaram por checagens automáticas e continuam com **`reviewed: false`**. O tutorial não pede que você preencha manualmente 216 linhas para executar esse piloto.

O piloto original, com outro backend, obteve 32/34 acertos no teste sintético congelado e 20/24 em um ensaio exploratório posterior. São conjuntos pequenos, sintéticos e diferentes entre si. Também foram observadas urgências excessivas em pedidos sem prazo iminente. **O desempenho da nova rota ainda será medido**; esses números não representam desempenho com clientes reais nem garantem o mesmo resultado.

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

Os testes offline padrão usam a biblioteca padrão do Python; não treinam nem instalam bibliotecas de GPU. O teste opcional de integração real Hugging Face fica desativado por padrão:

```bash
PYTHONPATH=ft05 python -m unittest discover -s ft05/tests -v
```

No PowerShell, use `$env:PYTHONPATH = "ft05"` antes de `python -m unittest discover -s ft05/tests -v`. Quem só deseja fazer o exercício no Colab não precisa abrir um terminal.

Este projeto foi separado do FT05 de [`radar-model-router`](https://github.com/ofelipepeixoto/radar-model-router). O projeto de origem foi preservado. Consulte a licença e o registro de origem incluídos aqui antes de redistribuir.
