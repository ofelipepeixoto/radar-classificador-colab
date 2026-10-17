# Homologação Colab — 17/10/2026

Status: novo treino na GPU, verificação dos artefatos e dois testes na interface
concluídos. A reinstalação e a reabertura em uma VM nova ainda estão pendentes.
Não é release nem validação para atendimento real.

**Replicação por outro operador em uma segunda conta independente: ainda não
comprovada.** A execução técnica observada não encerra essa etapa de homologação.

## Escopo

Novo treino com corpus sintético do repositório separado. Nenhum adaptador do
piloto antigo ou arquivo privado do autor foi reutilizado. As saídas privadas
permanecem no Drive do operador; este registro contém somente evidências
sanitizadas. Revisão humana dos rótulos continua pendente.

## Ambiente observado

- Código de treino: `d1a35409c2244206e9b772e6807c1fd6b1dae650`.
- Manifesto do código: SHA-256
  `398b105cd99c15e765f317e2f8026fad26c7b60550b18ea4964d8e192516bb8e`.
- Imagem Colab: `2026.07`; Python `3.12.13`.
- GPU: Tesla T4, `15360 MiB` no nvidia-smi; driver `580.82.07`.
- Instalador pré-existente do Colab: pip `24.1.2`.
- Ambiente sem pacotes do sistema; criação `venv --without-pip` e gerenciamento
  por `pip --python <venv/bin/python>`.
- Lock de 94 distribuições: SHA-256
  `facc8ea3aabd2675082270a6c9ec4f139ced0ef6374f28574b8c99d809f1c182`.

## Dados conferidos no Colab

208 mensagens fictícias; treino 162, validação 12 e teste 34. `human_reviewed:false`.

- Corpus: `47fd9736804f27f977ea2b30c6833cd2019a7c3af00302def5971900e9a25b72`.
- Manifesto das partições: `24f19b1d5616c4e11c095c5e33103cd3795753adaeebcbd7dc41670dbdc69941`.

## Falhas encontradas e corrigidas antes do treino

1. `ensurepip` ausente na imagem Colab: a criação agora usa o caminho oficial
   sem pip no venv. Nenhum pacote do kernel foi substituído.
2. `SFTConfig` atual não aceita `save_safetensors`: parâmetro removido; a
   configuração real compartilhada com produção passou pela integração CPU.
   O salvamento final PEFT continua exigindo safetensors.
3. BF16 exige suporte nativo; emulação não deve escolher BF16 para T4.
4. Processos auxiliares usam `-B`; o teste de importação trabalha em cópia
   temporária para não gerar bytecode no código conferido.
5. O painel encerrado pode ser reaberto na mesma execução.
6. Sucesso de treino exige passos concluídos e perdas/gradientes observados
   finitos; métricas e hash são persistidos no manifesto.

93 testes offline: 92 aprovados e 1 integração opcional omitida. A integração
HF CPU real foi executada separadamente com sucesso. Ela prepara dados com
TRL e verifica backprop LoRA/save/reload; não executa o treino TRL/CUDA.
Os [checks remotos do código treinado](https://github.com/ofelipepeixoto/radar-classificador-colab/actions/runs/39065079145)
foram concluídos com sucesso.

## Instalação e treino concluídos

- Instalação de 94 pacotes e pip check concluídos; operação CUDA FP16 16x16
  executada com resultado 4096.0. O gate foi ajustado ao valor utilizável medido
  da T4: 15.637.086.208 bytes (~14,56 GiB); novo piso 14 GiB.
- O gate integrado passou e o treinamento TRL foi executado na GPU.
- Pós-verificação: 63 passos de treino e 63 registros de gradiente finitos;
  `training_loss: 1.069351` e `eval_loss: 0.683872`.
- Pesos e selo de integridade conferidos; hashes do corpus e das partições
  permaneceram inalterados.
- As previsões cobrem os 34 IDs do teste congelado. As métricas foram
  recalculadas a partir dessas previsões e coincidiram com o relatório salvo.

## Resultado sintético do novo treino

| Métrica no teste congelado de 34 casos | Base sem adaptador | Novo adaptador |
| --- | --- | --- |
| Acerto exato de categoria + urgência | 0/34 (0%) | 30/34 (88,2%) |
| JSON válido | 0% | 100% |
| Macro F1 de categoria | 0 | 0,9064 |
| Macro F1 de urgência | 0 | 0,9696 |

Esta é uma execução nova com Transformers + PEFT + TRL. O resultado de
88,2% é diferente dos 94,1% (32/34) do piloto antigo com outro backend;
nenhum dos números comprova qualidade em atendimento real. Não ajustar
rótulos, hiperparâmetros ou retreinar usando os erros do teste congelado
como guia. A revisão humana e a avaliação independente continuam pendentes.

## Interface verificada na sessão do treino

| Mensagem fictícia | Resposta | Tempo observado |
| --- | --- | --- |
| Gostaria de saber quais formas de pagamento estão disponíveis para uma compra que farei no próximo mês. | `pagamento / baixa` | 3.273 ms, primeira classificação |
| Meu pedido precisa chegar hoje, antes das 17h, e o rastreio ainda não mostra saída para entrega. | `entrega / alta` | 870 ms |

As duas respostas passaram pela validação de formato. Seus rótulos eram
expectativas exploratórias, sem revisão independente; não formam uma métrica
de qualidade. Os tempos pertencem à sessão observada, não são uma garantia
de latência.

## Encerramento e etapa ainda pendente

O worker foi encerrado. A tela **Gerenciar sessões → Nenhuma sessão ativa**
confirmou o encerramento da VM original do Colab. Uma outra cópia de
`TESTAR-ADAPTADOR.ipynb` recebeu uma VM nova, na imagem `2026.07`, com Python
`3.12.13` e GPU T4. O código `d1a35409c2244206e9b772e6807c1fd6b1dae650`
foi obtido e os testes passaram nessa VM.

A montagem do Drive falhou por timeout no fluxo de autenticação Google,
terminando em `ValueError: mount failed`. Uma nova tentativa teve o mesmo
resultado. A janela do Google apresentou uma confirmação **Continuar** que
exige interação manual; essa etapa foi solicitada ao operador e continua
pendente. Nenhuma identidade de conta ou URL privada foi incluída neste registro.

Ainda falta concluir a montagem do Drive, instalar novamente o ambiente nessa VM, verificar os artefatos
salvos e reabrir o mesmo adaptador pela interface, sem retreino. Só depois
desse resultado o fluxo de reabertura poderá ser considerado homologado.
Isso permanece separado da replicação por outro operador/segunda conta,
que ainda não foi comprovada.
Continuar um treinamento interrompido a partir de checkpoint não faz parte
deste teste e não está implementado.

A issue #5 acompanha o risco residual upstream Accelerate. Não declarar
biblioteca integralmente corrigida nem pontuação sintética como validação real.
