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
- A retomada aceita o adaptador produzido por este fluxo com os registros esperados. O adaptador legado privado `run-002-g4` não é importado automaticamente.
- A interface de widgets fica no notebook e não abre um servidor público ou link externo de inferência.

Hashes permitem detectar mudanças nos bytes. Eles não autenticam arquivos recebidos de desconhecidos, não revisam o conteúdo dos dados e não comprovam a qualidade do modelo. Use somente artefatos da sua própria execução ou de uma procedência que você tenha verificado.

As versões de dependências são fixadas para reproduzir a configuração declarada do piloto. Fixar versões não significa que todos os pacotes foram auditados ou que serão compatíveis com qualquer runtime. Mudanças de dependência devem ser revisadas e testadas em sessão limpa.

## Limites de uso

O corpus permanece `reviewed: false`. Não publique afirmações de validação humana ou atendimento real com base neste exercício. O modelo pode devolver JSON correto com categoria ou urgência erradas; já houve superpriorização em situações sem prazo iminente.

As mensagens digitadas são entradas para classificação. O fluxo não deve executar comandos, acessar links, fazer pagamentos ou realizar cancelamentos com base no texto gerado. Qualquer integração futura exige uma avaliação própria.

## Como relatar uma vulnerabilidade

No GitHub, use a opção **Report a vulnerability** na aba **Security**, quando disponível. Se ela não estiver disponível, abra uma issue apenas com um resumo sem dados sensíveis e peça um canal privado para os detalhes. Não publique tokens, arquivos do Drive ou uma exploração que exponha terceiros.

Inclua a revisão afetada, etapa, impacto e reprodução mínima com dados fictícios. Não há prazo de resposta garantido. Para um token que já tenha vazado, revogue-o no serviço de origem; apagar a mensagem ou o commit não basta.

Para erros de classificação, instalação ou interface sem impacto de segurança, abra uma issue comum com o diagnóstico sanitizado.

## Aviso de dependência conhecido

A consulta de 17/10/2026 encontrou [GHSA-379c-qx7v-6h59 / CVE-2026-66007](https://github.com/advisories/GHSA-379c-qx7v-6h59) em `datasets==4.8.5`: metadados `file_name` de carregadores baseados em pastas podem ler arquivos fora do diretório. A versão corrigida informada é `5.0.1`, mas a Unsloth `2026.10.6` exige `datasets<5.0.0`. Não é seguro forçar uma instalação incompatível ou ignorar `pip check`.

O fluxo daqui usa somente `Dataset.from_dict` com texto sintético validado em memória. Não usa `load_dataset`, builders de pastas, `save_to_disk` ou `push_to_hub`. Assim, o caminho descrito no aviso não está exposto pelo exercício. Essa restrição é uma mitigação de escopo, não uma correção da biblioteca. A troca por uma combinação corrigida e compatível permanece pendente antes de anunciar uma versão estável; não expanda a entrada para datasets externos nessa configuração.

O scanner consultou cinco das seis dependências diretas; a variante CUDA de Torch e o lock transitivo completo não foram auditados. Não existe conclusão de “zero vulnerabilidades”.
