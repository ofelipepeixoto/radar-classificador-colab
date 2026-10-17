# Plano de implementação e homologação

## Objetivo

Uma pessoa abre um notebook, treina o próprio adaptador na GPU do Colab, salva no próprio Drive e testa mensagens fictícias por uma interface simples. Em outro dia, reabre o adaptador sem repetir o treinamento.

O repositório contém somente o material necessário ao FT05. O `radar-model-router` é preservado; os arquivos privados e o adaptador antigo do autor não são migrados.

## Estado das evidências

| Evidência | Alcance |
| --- | --- |
| Piloto de origem executado pelo autor | Treino sintético e inferência em Colab com GPU RTX PRO 6000 Blackwell, conforme saídas relatadas |
| Holdout de origem: 32/34 | Resultado sintético pequeno; não é validação real |
| Ensaio exploratório posterior: 20/24 | Criado após observar erros; indica superpriorização e não é avaliação independente |
| Interface pessoal usada no navegador | Respostas de categoria/urgência relatadas pelo autor; GPU no Colab |
| Novo fluxo separado | Execução completa em segunda conta e pasta nova ainda pendente |

O relatório de auditoria do novo repositório deve apresentar separadamente os testes realmente executados e as etapas que exigem Colab. Não há treinamento em GPU executado por esta documentação.

## Etapas

| Etapa | Entrega | Critério de conclusão |
| --- | --- | --- |
| Separação | Código, testes, corpus sintético e documentos do FT05 | Origem registrada, licença mantida e ausência de arquivos privados |
| Simplificação | `COMECE-AQUI.ipynb` e `TESTAR-ADAPTADOR.ipynb` | Instruções em português, ações desligadas por padrão, interface sem servidor externo |
| Auditoria offline | Testes, revisão de caminhos, persistência, integridade e conteúdo público | Falhas concretas corrigidas e evidência registrada |
| Dependências corrigidas | Resolver conflito entre Unsloth `<5.0.0` e correção Datasets `5.0.1` | Conjunto compatível instalado, auditado e testado na GPU; não forçar versões incompatíveis |
| Homologação Colab | Segunda conta autorizada e pasta nova | Todas as etapas abaixo concluídas com evidência |
| Versão para comunidade | Tag e instruções ligadas à revisão homologada | Links conferidos e limitações visíveis |
| Evolução | Melhorias por PRs pequenas | Preservar avaliação, privacidade e reprodutibilidade |

## Roteiro de homologação no Colab

Use a conta de quem vai executar o exercício e uma pasta nova. Não copie `run-002-g4`, o corpus privado ou relatórios do Drive do autor.

1. Salvar uma cópia de `COMECE-AQUI.ipynb`; registrar conta de teste de forma privada, data, commit, Python e GPU.
2. Obter o código e concluir os testes offline pelo notebook.
3. Montar o próprio Drive e confirmar o destino novo.
4. Preparar o corpus; conferir 208 exemplos, split 162/12/34 e revisão humana falsa.
5. Instalar explicitamente as dependências; registrar versões e necessidade de reinício.
6. Confirmar o uso de GPU e iniciar uma única execução nova de treino.
7. Conferir adaptador, relatório, metadados e integridade no Drive.
8. Abrir a interface e testar mensagens fictícias sem gravar dados reais no notebook.
9. Encerrar o runtime.
10. Abrir `TESTAR-ADAPTADOR.ipynb` em sessão nova; restaurar o ambiente e carregar o mesmo adaptador após as verificações.
11. Testar a interface novamente, confirmar que não ocorreu retreino e encerrar a sessão.
12. Registrar falhas, ajustes e etapas concluídas. Só então fixar uma tag para o tutorial homologado.

Esses passos homologam o fluxo técnico. Acurácia em atendimento real exige outro protocolo, dados representativos e rótulos revisados independentemente.

## Melhorias posteriores

- Reduzir dúvidas de instalação a partir de falhas reproduzidas em sessões limpas.
- Explicar a diferença entre urgência baixa e média com critérios consistentes.
- Avaliar modelos ou configurações novas em experimentos separados, com orçamento e métricas definidos antes do teste.
- Planejar suporte a outras classes e domínios sem alterar silenciosamente o exercício existente.
- Preparar avaliação real somente após definir autorização dos dados, anonimização, revisão independente e condições de uso.
