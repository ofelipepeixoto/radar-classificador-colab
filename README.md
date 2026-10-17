# Radar Classificador Colab

Laboratório educacional em português para criar e testar **seu próprio classificador LoRA** com a GPU do Google Colab e seu Google Drive. No computador, basta usar o navegador. O exemplo classifica mensagens fictícias por categoria e urgência; não conversa com clientes.

**Estado: prévia experimental em revisão.** O material executável está na [branch de homologação](https://github.com/ofelipepeixoto/radar-classificador-colab/tree/fix/safe-training-dependencies), dentro da [PR #4](https://github.com/ofelipepeixoto/radar-classificador-colab/pull/4). A PR #4 depende da [PR #1](https://github.com/ofelipepeixoto/radar-classificador-colab/pull/1). As duas estão em rascunho, sem merge ou tag estável. Portanto, a página `main` ainda não contém os notebooks.

## Começar pela prévia

1. Leia o [guia passo a passo para iniciantes](https://github.com/ofelipepeixoto/radar-classificador-colab/blob/fix/safe-training-dependencies/docs/COMECE-AQUI.md).
2. Abra [COMECE-AQUI no Colab](https://colab.research.google.com/github/ofelipepeixoto/radar-classificador-colab/blob/fix/safe-training-dependencies/COMECE-AQUI.ipynb), salve uma cópia em sua conta e escolha uma pasta nova no **seu** Drive.
3. Execute as células na ordem, com Python 3.12 e GPU compatível. Confira cada etapa antes de autorizar a instalação e o treinamento. O guia explica versões, custo variável e erros comuns.
4. Depois de salvar seu adaptador, use [TESTAR-ADAPTADOR no Colab](https://colab.research.google.com/github/ofelipepeixoto/radar-classificador-colab/blob/fix/safe-training-dependencies/TESTAR-ADAPTADOR.ipynb) em outra sessão para reabri-lo sem retreinar.

Cada participante cria seu adaptador com corpus sintético público; o projeto não distribui pesos ou arquivos privados do Drive do autor. Não há servidor público para inferência.

## O que a evidência comprova

| Verificação | Estado |
| --- | --- |
| Código do GitHub obtido e conferido pelo Colab | Concluída na conta do operador |
| Instalação isolada, CUDA e treino T4 de 63 passos | Concluídos; adaptador salvo e arquivos conferidos |
| Adaptador reaberto e testado em **outra VM** | Concluído pelo mesmo operador, sem retreino |
| Execução por **outra pessoa em segunda conta** | Pendente na [issue #2](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/2) |
| Revisão das PRs e link estável | Pendentes na [PR #1](https://github.com/ofelipepeixoto/radar-classificador-colab/pull/1), [PR #4](https://github.com/ofelipepeixoto/radar-classificador-colab/pull/4) e [issue #3](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/3) |
| Revisão humana dos rótulos e avaliação real | Pendentes na [issue #6](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/6) |

O novo teste sintético registrou **30/34 acertos e 100% de JSON válido**. Esses números não validam uso em atendimento real. Veja o [registro da execução](https://github.com/ofelipepeixoto/radar-classificador-colab/blob/fix/safe-training-dependencies/docs/HOMOLOGACAO-COLAB-2026-10-17.md), os [checks aprovados](https://github.com/ofelipepeixoto/radar-classificador-colab/actions/runs/39068101790) e a [auditoria de segurança](https://github.com/ofelipepeixoto/radar-classificador-colab/blob/fix/safe-training-dependencies/docs/AUDITORIA.md). Há um [risco residual de Accelerate, issue #5](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/5).

Você pode divulgar a **prévia para estudo e colaboração**, com o status experimental e os limites acima. Não a anuncie como versão estável ou classificador aprovado para clientes.

O projeto FT05 foi separado de [`radar-model-router`](https://github.com/ofelipepeixoto/radar-model-router), que permanece preservado. Código original de Carlos Felipe, com assistência de IA. Licença MIT; modelos, bibliotecas e serviços mantêm seus próprios termos.
