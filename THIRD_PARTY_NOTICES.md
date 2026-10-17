# Origem, créditos e licenças

Extraído e adaptado do FT05 de Carlos Felipe em `ofelipepeixoto/radar-model-router`,
com assistência de IA. O código original permanece disponível e intacto.
Os commits consultados estão em [SOURCE-PROVENANCE.json](SOURCE-PROVENANCE.json).
A licença MIT deste repositório cobre este código e os exemplos sintéticos originais;
não relicencia pesos, bibliotecas ou serviços externos.

O FT05 começou como implementação de um exercício descrito no Guia FT05,
Eduardo Cavalcanti, 07/10/2026, consultado privadamente na origem.
Esse guia/PDF não é redistribuído aqui e não é necessário para executar o código.
O corpus é fictício, criado com assistência de IA; os rótulos não têm revisão humana.

## Dependências obtidas pelo usuário, sem pesos redistribuídos

- Modelo base: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
  revisão `5fee7c4ed634dc66c6e318c8ac2897b8b9154536` na configuração. A página
  oficial declara Apache-2.0. Verifique o arquivo LICENSE da revisão baixada.
- [Unsloth](https://github.com/unslothai/unsloth/blob/main/LICENSE): o núcleo
  declara Apache-2.0. Componentes opcionais têm termos próprios; este projeto
  não inclui Unsloth Studio nem seu código de interface.
- PyTorch, Transformers, TRL, Datasets, PEFT e ipywidgets mantêm suas próprias
  licenças e dependências. São instalados no runtime do usuário, quando necessário.
- Google Colab e Drive são serviços externos sujeitos aos termos do Google.
  Não são software hospedado por este projeto e não há promessa de gratuidade.

Referências oficiais consultadas em 17/10/2026. Licenças e artefatos concretos
devem ser reconferidos antes de redistribuir um adaptador ou empacotar dependências.
Não foi copiado código da política Jev, do roteador, do estudo Laya ou do n8n.
