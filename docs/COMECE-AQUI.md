# Seu primeiro classificador no Colab

Você vai treinar um adaptador com mensagens fictícias e depois digitar novos textos em uma interface. Treino e inferência acontecem na GPU do Colab. Seu computador mostra a página no navegador.

**Este fluxo ainda está em homologação.** A execução completa do novo repositório em segunda conta e pasta nova permanece pendente. Os resultados anteriores pertencem ao piloto de origem.

## 1. Abra e salve sua cópia

1. Abra [COMECE-AQUI.ipynb](../COMECE-AQUI.ipynb) no Colab.
2. Use **Arquivo → Salvar uma cópia no Drive**.
3. Dê um nome fácil de reconhecer, por exemplo `Meu classificador — treino.ipynb`.
4. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, selecione uma GPU de classe 16 GB ou maior disponível na sua conta e confira Python 3.12.
5. Confira seu saldo e a disponibilidade da sessão. A receita usa Torch com CUDA 12.9; a compatibilidade com o driver e a GPU precisa passar pelas verificações. Uma GPU de 16 GB pode aparecer com cerca de 15 GiB, e passar pelo limite mínimo não comprova que todo o treino funcionará.

Execute as células na ordem apresentada. **Não use “Executar tudo”.** Leia a explicação da etapa e confira sua saída antes de seguir.

## 2. Escolha uma pasta nova

Na configuração do notebook, escolha uma pasta de projeto e um nome de execução. Exemplo:

```python
PASTA_PROJETO = 'RadarClassificador/meu-piloto-001'
NOME_EXECUCAO = 'run-001'
```

O destino será `Meu Drive/RadarClassificador/meu-piloto-001`. Na primeira execução, essa pasta deve ser nova. Preserve cada tentativa de treino; use outro nome de execução quando uma tentativa anterior já tiver criado arquivos.

As ações começam desligadas. Cada etapa tem sua própria caixa de seleção: marque a opção e execute **aquela célula**. Não é necessário voltar à configuração. A instalação exige `INSTALAR_DEPENDENCIAS`; o treino exige `TREINAR`, `ORCAMENTO_GPU_CONFIRMADO` e `DEPENDENCIAS_REVISADAS`. O padrão `TREINAR = False` mantém o treinamento desligado. Não execute a célula de treino novamente depois de concluir.

## 3. Siga as etapas do notebook

| Ordem | O que fazer | Confira antes de avançar |
| --- | --- | --- |
| Ambiente | Execute a verificação de Python e GPU | A GPU aparece; versão e hardware estão registrados |
| Código e testes | Marque `OBTER_CODIGO` e execute a célula | Revisão exibida e testes terminam com `OK` |
| Drive | Marque `MONTAR_DRIVE` e confira a pasta | O destino aponta para o seu Drive |
| Corpus | Marque `PREPARAR_CORPUS` | 208 exemplos; 162/12/34; revisão humana falsa |
| Dependências | Marque `CONFERIR_DEPENDENCIAS`; use `REVISAR_PLANO` para conferir o plano fixado e só depois `INSTALAR_DEPENDENCIAS` na célula seguinte | Ambiente isolado criado; versões, hashes, `pip check` e CUDA conferidos |
| Treinamento | Marque `TREINAR` e as confirmações após as verificações | Artefatos são salvos na pasta nova da execução |
| Interface | Marque `ABRIR_INTERFACE` e teste mensagens fictícias | Categoria, urgência e tempo aparecem após o clique |

Se uma etapa falhar, pare nessa etapa. Copie a mensagem de erro para diagnosticar; não desative as verificações nem altere hashes ou rótulos para fazê-la passar.

### Como interpretar o corpus

O projeto gera 208 exemplos a partir de 216 candidatos públicos, com exclusão automática de oito casos ambíguos. Todos os exemplos usados continuam `reviewed: false`. Essa preparação é suficiente para executar o **exercício sintético**, sem fingir que houve aprovação humana.

O teste congelado deve permanecer separado do treino. As mensagens digitadas depois são observações exploratórias; não contam como uma avaliação independente de qualidade real.

### Sobre as dependências

O notebook usa Transformers + PEFT + TRL para treinar LoRA em 16 bits, sem Unsloth e sem quantização de 4/8 bits. As versões diretas são:

| Pacote | Versão |
| --- | --- |
| PyTorch | `2.13.0+cu129` |
| Transformers | `5.13.1` |
| TRL | `1.15.0` |
| Datasets | `5.0.1` |
| PEFT | `0.19.1` |
| Accelerate | `1.15.0` |
| ipywidgets | `8.1.8` |
| setuptools | `83.0.0` |

A instalação cria `/content/radar-classificador-env-hf`: um ambiente Python separado, **sem herdar os pacotes já instalados no Colab**. O arquivo `requirements-colab-lock.txt` fixa 94 distribuições com hashes para Python 3.12/Linux x86_64. O plano já vem resolvido: `REVISAR_PLANO` mostra suas versões e hashes sem baixar a distribuição CUDA. A resolução foi conferida sem GPU; instalar e executar na GPU ainda são etapas de homologação.

Use as células fornecidas para revisar, instalar e conferir o ambiente. Elas criam o ambiente sem pip próprio e usam o pip disponível no Colab para gerenciar **somente o Python isolado**. Isso contorna a ausência de `ensurepip` em algumas sessões, mantendo as mesmas 94 dependências fixadas.

A instalação baixa apenas as URLs diretas do lock e exige os hashes; os pacotes do kernel Colab não são alterados e não precisam de reinício por essa instalação. Não rode `%pip install` no kernel nem `python -m pip` pelo Python isolado para consertá-lo: as células já direcionam instalação e conferência ao ambiente correto. Um aviso antigo do IPython sobre `jedi` no kernel não deve levar a alterar os pacotes do treinamento.

Na verificação de GPU, a receita seleciona BF16 somente quando há suporte nativo; a T4 deve usar FP16. Confira a precisão exibida antes de treinar. A instalação só está concluída quando a célula confirma versões, `pip check` e CUDA; montar o Drive e preparar o corpus não comprova essas etapas.

Datasets `5.0.1` remove a pendência da versão antiga usada no piloto. Há um risco residual em Accelerate para checkpoints fragmentados; o projeto restringe a entrada à base fixada e ao adaptador local selado do seu próprio treino. Isso não significa que Accelerate foi corrigido; veja [SECURITY.md](../SECURITY.md#risco-residual-em-accelerate). Não carregue datasets ou pesos externos neste fluxo.

## 4. Teste pela interface

Digite apenas o texto da mensagem e clique em **Classificar**. Exemplo fictício:

> Minha encomenda deve chegar hoje e preciso recebê-la antes do fim do dia.

A interface exibirá uma categoria e uma urgência. Ela não responde ao cliente nem realiza a solicitação. O modelo pode errar, inclusive aumentar a prioridade de pedidos sem prazo imediato.

O modelo fica carregado em um processo do ambiente isolado para os próximos cliques. **O texto digitado na interface não gera treino.** A interface não publica um endereço de acesso externo e não salva um histórico automaticamente. Saídas de outras células podem permanecer gravadas no próprio notebook; revise-as antes de compartilhar sua cópia.

Para mudar para outro adaptador depois de carregar a interface, reinicie a sessão.
Isso evita mostrar previsões do modelo anterior após alterar os nomes da configuração.

## 5. Confira o que foi salvo

No Drive, preserve a pasta inteira de sua execução, incluindo o adaptador, `run.json`, relatórios e arquivos de integridade gerados pelo notebook. Preserve também corpus e split congelado. Renomear o arquivo `.ipynb` é permitido; mover ou renomear essas pastas exige ajustar a configuração.

O notebook para reabrir o adaptador aceita os artefatos produzidos por este fluxo e confere sua procedência, backend e integridade. Ele não importa automaticamente o adaptador privado antigo `run-002-g4` do autor. Os hashes detectam mudanças em arquivos; não certificam que um adaptador desconhecido é seguro ou correto.

## 6. Volte a testar em outro dia

1. Abra uma cópia de [TESTAR-ADAPTADOR.ipynb](../TESTAR-ADAPTADOR.ipynb) na mesma conta que contém seus resultados.
2. Escolha uma GPU compatível e informe a mesma `PASTA_PROJETO` e o mesmo `NOME_EXECUCAO` que você criou.
3. Siga as células de ambiente, código, Drive, dependências e verificação do adaptador. O ambiente isolado fica em `/content` e deve ser recriado quando uma sessão nova não o tiver.
4. Abra a interface quando todas as verificações passarem. Não habilite treino para apenas testar.

Ao reiniciar o Colab, a memória da sessão e o ambiente em `/content` podem desaparecer. Arquivos persistidos no Drive continuam disponíveis. Uma pasta de código já existente deve ser validada e reutilizada conforme o notebook; não crie uma nova tentativa de treino só para restaurar imports.

Aqui, **reabrir** significa carregar um adaptador já concluído para fazer inferência. Continuar um treinamento interrompido a partir de checkpoint não está implementado. Se um treino falhar, preserve a tentativa e diagnostique antes de iniciar outra com nome novo.

## Se algo não funcionar

| Saída ou sintoma | Próxima ação |
| --- | --- |
| GPU não detectada | Selecione GPU nas configurações e rode a verificação de novo |
| GPU abaixo do limite ou incompatível com CUDA | Use uma sessão compatível; não desative a verificação de memória ou driver |
| Código ausente | Execute a etapa de obtenção do código antes das importações |
| Drive indisponível | Execute a montagem e confira a conta e a `PASTA_PROJETO` |
| `PackageNotFoundError` ou Python isolado ausente | Conclua a instalação pela célula fornecida; não instale no kernel com `%pip` |
| Erro antigo sobre `ensurepip` | Use a cópia atualizada do notebook e repita a célula de instalação; o fluxo atual cria o ambiente sem pip próprio |
| Versões divergentes | Preserve a saída e confira o diagnóstico antes de continuar |
| Pasta de execução já existe | Preserve a tentativa; investigue o erro e escolha novo nome se for treinar novamente |
| Hash ou procedência não confere | Não edite o manifesto; confira os caminhos e use os arquivos originais da mesma execução |
| JSON inválido ou prioridade inesperada | Registre o comportamento como erro do modelo; a interface funcionar não garante o rótulo correto |
| Widgets não aparecem | Confira a saída da célula de interface e se as verificações anteriores passaram |
| Interface informa que o processo foi encerrado | Execute novamente a célula `ABRIR_INTERFACE` para reabrir o mesmo adaptador; isso não inicia treino |

Ao abrir uma issue, inclua a etapa, o commit, as versões, GPU e erro completo após remover nomes, caminhos pessoais e tokens. Não envie seu Drive inteiro ou pesos do modelo.

## Ao terminar

Salve a cópia do notebook e confira seus arquivos no Drive. Use **Ambiente de execução → Desconectar e excluir ambiente de execução** para liberar a sessão de GPU. Encerrar o runtime não exclui os resultados persistidos no Drive.
