# Seu primeiro classificador no Colab

Você vai treinar um adaptador com mensagens fictícias e depois digitar novos textos em uma interface. Treino e inferência acontecem na GPU do Colab. Seu computador mostra a página no navegador.

**Este fluxo ainda está em homologação.** A execução completa do novo repositório em segunda conta e pasta nova permanece pendente. Os resultados anteriores pertencem ao piloto de origem.

## 1. Abra e salve sua cópia

1. Abra [COMECE-AQUI.ipynb](../COMECE-AQUI.ipynb) no Colab.
2. Use **Arquivo → Salvar uma cópia no Drive**.
3. Dê um nome fácil de reconhecer, por exemplo `Meu classificador — treino.ipynb`.
4. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, selecione uma GPU disponível na sua conta.
5. Confira seu saldo e a disponibilidade da sessão. A escolha da GPU não é uma garantia de compatibilidade.

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
| Dependências | Marque `CONFERIR_DEPENDENCIAS`; se faltarem pacotes, marque `RESOLVER_INSTALACAO` para ver o plano e só depois `INSTALAR_DEPENDENCIAS` na célula seguinte | Versões coincidem e `pip check` termina com `OK` |
| Treinamento | Marque `TREINAR` e as confirmações após as verificações | Artefatos são salvos na pasta nova da execução |
| Interface | Marque `ABRIR_INTERFACE` e teste mensagens fictícias | Categoria, urgência e tempo aparecem após o clique |

Se uma etapa falhar, pare nessa etapa. Copie a mensagem de erro para diagnosticar; não desative as verificações nem altere hashes ou rótulos para fazê-la passar.

### Como interpretar o corpus

O projeto gera 208 exemplos a partir de 216 candidatos públicos, com exclusão automática de oito casos ambíguos. Todos os exemplos usados continuam `reviewed: false`. Essa preparação é suficiente para executar o **exercício sintético**, sem fingir que houve aprovação humana.

O teste congelado deve permanecer separado do treino. As mensagens digitadas depois são observações exploratórias; não contam como uma avaliação independente de qualidade real.

### Sobre as dependências

As versões de referência são Unsloth `2026.10.6`, PyTorch `2.11.0+cu128`, Transformers `5.13.1`, TRL `1.15.0`, Datasets `4.8.5` e PEFT `0.19.1`. Elas reproduzem a configuração declarada do piloto de origem. Existe um aviso conhecido em Datasets para carregadores de pastas que não são usados neste exercício. A correção upstream em `5.0.1` conflita com a exigência `<5.0.0` da Unsloth fixada; veja os limites e o plano em [AUDITORIA.md](AUDITORIA.md). Não carregue datasets externos ou pastas de imagens/áudio. O novo fluxo ainda precisa confirmar sua instalação em uma sessão limpa; não há garantia para runtimes futuros.

Uma instalação pode solicitar reinício da sessão. Nesse caso, retome pelas instruções abaixo e confira as versões outra vez antes de carregar ou treinar o modelo.

O arquivo `requirements-lock-colab.txt` no seu Drive registra o ambiente instalado.
As seis versões diretas não fixam todas as dependências transitivas. Confira a saída
de `pip check` antes de qualquer ação GPU. No piloto de origem houve um conflito
`ipython ... requires jedi, which is not installed`: se esse for o único erro,
revise a dependência ausente na versão do IPython da sessão e instale-a explicitamente
com `%pip install jedi` em uma célula separada. Depois repita a etapa 5 e confirme
`pip check: OK`. Não use essa correção para conflitos diferentes nem rebaixe o
ambiente inteiro. Registre a versão efetiva no lock; essa recuperação ainda precisa
ser confirmada em uma sessão nova do projeto.

## 4. Teste pela interface

Digite apenas o texto da mensagem e clique em **Classificar**. Exemplo fictício:

> Minha encomenda deve chegar hoje e preciso recebê-la antes do fim do dia.

A interface exibirá uma categoria e uma urgência. Ela não responde ao cliente nem realiza a solicitação. O modelo pode errar, inclusive aumentar a prioridade de pedidos sem prazo imediato.

O modelo fica carregado na sessão para os próximos cliques. **O texto digitado na interface não gera treino.** A interface não publica um endereço de acesso externo e não salva um histórico automaticamente. Saídas de outras células podem permanecer gravadas no próprio notebook; revise-as antes de compartilhar sua cópia.

Para mudar para outro adaptador depois de carregar a interface, reinicie a sessão.
Isso evita mostrar previsões do modelo anterior após alterar os nomes da configuração.

## 5. Confira o que foi salvo

No Drive, preserve a pasta inteira de sua execução, incluindo o adaptador, `run.json`, relatórios e arquivos de integridade gerados pelo notebook. Preserve também corpus e split congelado. Renomear o arquivo `.ipynb` é permitido; mover ou renomear essas pastas exige ajustar a configuração.

O notebook de retomada aceita os artefatos produzidos por este fluxo e confere sua procedência e integridade. Ele não importa automaticamente o adaptador privado antigo `run-002-g4` do autor. Os hashes detectam mudanças em arquivos; não certificam que um adaptador desconhecido é seguro ou correto.

## 6. Volte a testar em outro dia

1. Abra uma cópia de [TESTAR-ADAPTADOR.ipynb](../TESTAR-ADAPTADOR.ipynb) na mesma conta que contém seus resultados.
2. Escolha uma GPU compatível e informe a mesma `PASTA_PROJETO` e o mesmo `NOME_EXECUCAO` que você criou.
3. Siga as células de ambiente, código, Drive, dependências e verificação do adaptador.
4. Abra a interface quando todas as verificações passarem. Não habilite treino para apenas testar.

Ao reiniciar o Colab, a memória da sessão e os pacotes podem desaparecer. Arquivos persistidos no Drive continuam disponíveis. Uma pasta de código já existente deve ser validada e reutilizada conforme o notebook; não crie uma nova tentativa de treino só para restaurar imports.

## Se algo não funcionar

| Saída ou sintoma | Próxima ação |
| --- | --- |
| GPU não detectada | Selecione GPU nas configurações e rode a verificação de novo |
| Código ausente | Execute a etapa de obtenção do código antes das importações |
| Drive indisponível | Execute a montagem e confira a conta e a `PASTA_PROJETO` |
| `PackageNotFoundError` para Unsloth ou TRL | Conclua a instalação da sessão; reinicie apenas se solicitado |
| Versões divergentes | Preserve a saída e confira o diagnóstico antes de continuar |
| Pasta de execução já existe | Preserve a tentativa; investigue o erro e escolha novo nome se for treinar novamente |
| Hash ou procedência não confere | Não edite o manifesto; confira os caminhos e use os arquivos originais da mesma execução |
| JSON inválido ou prioridade inesperada | Registre o comportamento como erro do modelo; a interface funcionar não garante o rótulo correto |
| Widgets não aparecem | Confira a saída da célula de interface e se as verificações anteriores passaram |

Ao abrir uma issue, inclua a etapa, o commit, as versões, GPU e erro completo após remover nomes, caminhos pessoais e tokens. Não envie seu Drive inteiro ou pesos do modelo.

## Ao terminar

Salve a cópia do notebook e confira seus arquivos no Drive. Use **Ambiente de execução → Desconectar e excluir ambiente de execução** para liberar a sessão de GPU. Encerrar o runtime não exclui os resultados persistidos no Drive.
