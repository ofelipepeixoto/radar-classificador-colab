# Homologação Colab — 17/10/2026

Status: execução em andamento; não é release nem validação para atendimento real.

## Escopo

Novo treino com corpus sintético do repositório separado. Nenhum adaptador do
piloto antigo ou arquivo privado do autor foi reutilizado. As saídas privadas
permanecem no Drive do operador; este registro contém somente evidências
sanitizadas. Revisão humana dos rótulos continua pendente.

## Ambiente observado

- Código de treino: `7cd591ec3f5cc2180135eddb2d80e3d7d1cafe55`.
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
Checks remotos do código acima:
https://github.com/ofelipepeixoto/radar-classificador-colab/actions/runs/39064476951

## Pendências desta execução

- Instalação de 94 pacotes e pip check concluídos; operação CUDA FP16 16x16
  executada com resultado 4096.0. O gate foi ajustado ao valor utilizável medido
  da T4: 15.637.086.208 bytes (~14,56 GiB); novo piso 14 GiB.
- Confirmação do gate integrado depois desse ajuste.
- Treino TRL completo e métricas reais.
- Integridade dos pesos/comparação.
- Interface e reabertura em sessão nova sem retreino.

A issue #5 acompanha o risco residual upstream Accelerate. Não declarar
biblioteca integralmente corrigida nem pontuação sintética como validação real.
