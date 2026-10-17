# Segurança e privacidade

Este projeto é um exercício sintético de classificação. O runtime do Colab executa código e pode acessar o Drive montado. Use uma cópia de origem conhecida, confira a revisão exibida e revise as ações antes de habilitá-las.

## Conteúdo público e privado

**Público:** código, testes, documentação e mensagens sintéticas do exercício.

**Privado no seu Drive:** adaptador treinado, metadados da execução, relatórios e mensagens pessoais de teste. O repositório não precisa receber esses arquivos. Antes de compartilhar uma cópia do notebook, remova saídas que contenham textos ou caminhos pessoais.

Não envie tokens, chaves, credenciais, documentos ou mensagens de clientes ao corpus, às issues ou às PRs. O fluxo padrão não solicita chave de API ou pagamento por uma API externa; o uso de GPU e armazenamento depende da sua conta Google.

## Verificações do fluxo

- Instalação e treinamento exigem ativação explícita; `INSTALAR_DEPENDENCIAS` e `TREINAR` começam em `False` nas respectivas células.
- Cada pessoa escolhe uma pasta nova; uma tentativa existente deve ser preservada para diagnóstico.
- Corpus, split e artefatos passam por verificações de integridade antes de uso.
- A reabertura para inferência aceita o adaptador produzido pelo backend atual com os registros esperados. O adaptador legado privado `run-002-g4` não é importado automaticamente. Retomar treino de checkpoint não está implementado.
- A interface de widgets fica no notebook e não abre um servidor público ou link externo de inferência.
- As 94 distribuições do ambiente de treino são fixadas por URLs diretas e hashes, para Python 3.12/Linux x86_64. O instalador exige `--require-hashes` e `--no-index`, confere versões, executa `pip check` e verifica CUDA antes de liberar ações GPU.

Hashes permitem detectar mudanças nos bytes. Eles não autenticam arquivos recebidos de desconhecidos, não revisam o conteúdo dos dados e não comprovam a qualidade do modelo. Use somente artefatos da sua própria execução ou de uma procedência que você tenha verificado.

Treino e inferência usam Transformers + PEFT + TRL em `/content/radar-classificador-env-hf`, um ambiente sem `system-site-packages`. O kernel Colab não recebe esses pacotes. Esse isolamento evita herdar conflitos do kernel; **não é uma barreira de segurança** contra código malicioso e mantém as permissões de acesso do processo, inclusive ao Drive montado.

O ambiente é criado com `venv --without-pip`. O pip já disponível no kernel é chamado com `--python` apontando ao Python isolado, conforme a [documentação oficial do pip](https://pip.pypa.io/en/stable/topics/python-option/). Assim, não se depende de `ensurepip` no Colab e a instalação continua restrita ao destino isolado, com o mesmo lock. Os processos de inferência usam `-B` para não gerar bytecode no código conferido.

O treino só pode ser declarado concluído após as verificações de passos, perdas e métricas finitas; a seleção BF16 exige suporte nativo, usando FP16 quando ele não existe. Esses controles precisam passar na homologação real e não substituem avaliar a qualidade do adaptador.

Fixar versões e hashes não comprova que todos os pacotes são seguros ou que funcionarão em qualquer runtime. A resolução sem GPU não valida CUDA, driver, memória ou treinamento. Mudanças de dependência devem ser revisadas e testadas em sessão limpa.

## Limites de uso

O corpus permanece `reviewed: false`. Não publique afirmações de validação humana ou atendimento real com base neste exercício. O modelo pode devolver JSON correto com categoria ou urgência erradas; já houve superpriorização em situações sem prazo iminente.

As mensagens digitadas são entradas para classificação. O fluxo não deve executar comandos, acessar links, fazer pagamentos ou realizar cancelamentos com base no texto gerado. Qualquer integração futura exige uma avaliação própria.

## Como relatar uma vulnerabilidade

No GitHub, use a opção **Report a vulnerability** na aba **Security**, quando disponível. Se ela não estiver disponível, abra uma issue apenas com um resumo sem dados sensíveis e peça um canal privado para os detalhes. Não publique tokens, arquivos do Drive ou uma exploração que exponha terceiros.

Inclua a revisão afetada, etapa, impacto e reprodução mínima com dados fictícios. Não há prazo de resposta garantido. Para um token que já tenha vazado, revogue-o no serviço de origem; apagar a mensagem ou o commit não basta.

Para erros de classificação, instalação ou interface sem impacto de segurança, abra uma issue comum com o diagnóstico sanitizado.

## Datasets atualizado

A extração inicial fixava Datasets `4.8.5`, afetado por [GHSA-379c-qx7v-6h59](https://github.com/advisories/GHSA-379c-qx7v-6h59). A nova rota remove Unsloth e sua restrição `<5.0.0`, permitindo Datasets `5.0.1`, a versão corrigida informada no aviso. Esse tratamento é acompanhado na [issue #3](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/3).

O exercício continua usando apenas `Dataset.from_dict` com texto sintético validado em memória. A atualização não autoriza ampliar o fluxo para datasets externos, builders de pasta ou arquivos de clientes.

## Risco residual em Accelerate

[GHSA-4j2p-28q2-5m79 / CVE-2026-69112](https://github.com/advisories/GHSA-4j2p-28q2-5m79) trata de caminhos não validados em `weight_map` de índices de checkpoints fragmentados. O feed lista versões até `1.14.0` e nenhuma correção confirmada. A ausência de alerta OSV para `1.15.0` não comprova correção: a inspeção do wheel usado nesta rota ainda encontrou o caminho de código relevante.

A aplicação restringe o carregamento à base conhecida em revisão fixa, sem código remoto, e a um adaptador local do próprio treino: `adapter_config.json` e `adapter_model.safetensors`, com hashes conferidos. Índices de fragmentos, pesos pickle e o caminho de dispatch de checkpoints externos não são entradas permitidas. Isso reduz a exposição do exercício; **não corrige a biblioteca Accelerate**. Não carregue checkpoints de terceiros nem amplie formatos sem rever essa limitação.

Não há conclusão de “zero vulnerabilidades”. A [auditoria](docs/AUDITORIA.md) distingue resolução, inspeção de código, testes offline e a homologação GPU ainda pendente na [issue #2](https://github.com/ofelipepeixoto/radar-classificador-colab/issues/2).
