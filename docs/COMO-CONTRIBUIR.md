# Como contribuir

Obrigado por ajudar a tornar o exercício mais simples e reproduzível. A prioridade atual é concluir a homologação dos notebooks em uma segunda conta e pasta nova no Drive.

## Uma mudança por PR

1. Abra uma issue para mudanças maiores, como trocar modelo, classes, corpus ou protocolo de avaliação.
2. Crie um fork e uma branch para a alteração.
3. Faça uma mudança pequena e explique o problema que resolve.
4. Rode os testes offline e revise os arquivos enviados.
5. Abra uma PR com a evidência disponível e os limites da validação.

Boas primeiras contribuições: melhorar uma instrução confusa, reproduzir um erro com mensagem fictícia, tornar um diagnóstico acionável ou acrescentar um teste de uma falha real. Para segurança, siga [SECURITY.md](../SECURITY.md).

## Testes offline

Na raiz do repositório:

```bash
PYTHONPATH=ft05 python -m unittest discover -s ft05/tests -v
```

PowerShell:

```powershell
$env:PYTHONPATH = "ft05"
python -m unittest discover -s ft05/tests -v
```

Esses testes não instalam a stack de treino, não baixam pesos e não executam treino em GPU. Eles verificam o código que pode ser exercitado offline; o teste opcional de integração real fica desativado por padrão. Passarem não comprova instalação, CUDA ou inferência no Colab.

A rota atual usa Transformers + PEFT + TRL em ambiente isolado, com LoRA em 16 bits. Ao alterar dependências, mantenha coerentes `ft05/dependencies.py`, as versões diretas em `requirements-colab.txt` e o lock completo `requirements-colab-lock.txt`. O lock é específico para Python 3.12/Linux x86_64 e requer resolução e revisão de hashes. Não copie pacotes do kernel nem use `--system-site-packages` para contornar conflitos.

O teste real opcional usa um modelo Qwen minúsculo aleatório, local e CPU: verifica um passo de backward/AdamW em LoRA, gravação, reabertura, geração e preparação de dados pelo construtor `SFTTrainer`. **Não chama `SFTTrainer.train()`**, cujo caminho depende de Triton/CUDA nessa versão. O teste exige as bibliotecas instaladas em ambiente separado e pode ser executado com `RUN_HF_CPU_INTEGRATION=1` junto do comando de testes. Não baixa a base 1,5B, não substitui homologação na GPU e não mede a qualidade do classificador.

Quando mudar o notebook ou seu gerador, mantenha os dois consistentes e registre o que foi executado no Colab. Uma alteração somente documental não exige treinar novamente. Teste o caminho afetado pela alteração.

## O que incluir na PR

- Problema observado e mudança feita.
- Comando executado e resultado dos testes.
- Commit, Python, GPU e versões, se houve execução no Colab.
- Etapas ainda não verificadas.
- Resolução do lock, `pip check` do ambiente isolado e resultado da integração CPU, se a alteração envolver o backend.
- Exemplo fictício mínimo para reproduzir falhas do classificador.

Não inclua tokens, mensagens de clientes, notebook com saídas pessoais, pastas do Drive, pesos ou adaptadores. Limpe saídas antes de enviar notebooks. Consulte a licença e o registro de origem quando reutilizar código.

## Preserve a avaliação

O corpus atual é sintético e não possui aprovação humana. Não marque `reviewed: true` com base apenas em uma checagem automática.

Não ajuste o treino usando os erros do teste congelado ou dos ensaios exploratórios como guia para aumentar suas métricas. Uma proposta de novo corpus precisa explicar procedência, critérios de rótulo, separação entre treino e avaliação e prevenção de duplicatas. Mudanças de modelo e dependências precisam de uma execução verificável em ambiente limpo. Registre resultados do novo backend separadamente dos resultados históricos do piloto com Unsloth.

Não existe um requisito de preencher manualmente as 216 linhas antigas para contribuir com documentação ou executar o piloto sintético. Uma avaliação com dados reais é um trabalho separado e requer rótulos independentes.
