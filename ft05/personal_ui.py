"""Private in-notebook UI for repeated inference on a saved FT05 adapter.

No web server, sharing link, message persistence, or training. The caller
mounts their own Drive and verifies consent for the Colab GPU session.
"""
def launch(run_dir, corpus, frozen):
    """Load the pinned adapter once, then return a browser widget in Colab."""
    from personal_smoke import preflight, validate_messages, load_saved_adapter
    from ft05 import target
    from train import generate_response

    checked = preflight(run_dir, corpus, frozen, versions=True)
    import ipywidgets as widgets
    from IPython.display import display

    model, tokenizer, _ = load_saved_adapter(checked)

    box = widgets.Textarea(value='', placeholder='Digite uma mensagem fictícia...',
                           description='Mensagem:', layout=widgets.Layout(width='95%', height='110px'))
    button = widgets.Button(description='Classificar', button_style='primary')
    output = widgets.Output()

    def classify(_):
        button.disabled = True
        try:
            with output:
                output.clear_output()
                message = box.value.strip()
                if not 3 <= len(message) <= 2000:
                    print('Digite entre 3 e 2.000 caracteres.')
                    return
                try:
                    validate_messages([{'id': 'interface', 'text': message}])
                except ValueError as exc:
                    print('Mensagem rejeitada:', str(exc))
                    print('Use somente uma mensagem fictícia, sem dados pessoais ou segredos.')
                    return
                raw, elapsed_ms = generate_response(
                    model, tokenizer, [{'role': 'user', 'content': message}])
                try:
                    labels = target(raw)
                    print('Categoria:', labels['categoria'])
                    print('Urgência:', labels['urgencia'])
                except (ValueError, TypeError):
                    print('Resposta fora do formato esperado:', raw)
                print('Tempo:', round(elapsed_ms), 'ms')
                print('Teste pessoal sintético; não é decisão automática para clientes.')
        except Exception as exc:
            with output:
                output.clear_output()
                print('Falha na inferência:', type(exc).__name__, str(exc))
        finally:
            button.disabled = False

    button.on_click(classify)
    panel = widgets.VBox([box, button, output])
    display(panel)
    print('Interface pronta. O modelo fica carregado até encerrar o runtime.')
    return panel
