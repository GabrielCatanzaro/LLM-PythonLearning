"""Frontend (Gradio) para o usuario digitar o prompt e testar o modelo.

    python app.py --adapter modelos/base_lr3e-4_r8_e5
Abre em http://127.0.0.1:7860
"""
import argparse
import contextlib
import time

import gradio as gr

from gpu_utils import descrever, get_device
from llm_utils import MODELO_PADRAO, carregar_para_inferencia, carregar_tokenizer, gerar_stream

ap = argparse.ArgumentParser()
ap.add_argument("--modelo", default=MODELO_PADRAO)
ap.add_argument("--adapter", default="modelos/base_lr3e-4_r8_e5")
ap.add_argument("--porta", type=int, default=7860)
ap.add_argument("--gpu", type=int, default=None)
ap.add_argument("--permitir_cpu", action="store_true")
args = ap.parse_args()

device, backend, nome_gpu = get_device(args.gpu, args.permitir_cpu)
info_gpu = descrever(device, backend, nome_gpu)
print(info_gpu)

tok = carregar_tokenizer(args.modelo)
modelo, tem_adapter = carregar_para_inferencia(args.modelo, args.adapter, device)
if not tem_adapter:
    print(f"[aviso] adaptador nao encontrado em {args.adapter}; so o modelo base estara disponivel.")

OPCOES = ["Ajustado (LoRA)", "Base (sem ajuste)"] if tem_adapter else ["Base (sem ajuste)"]


def responder(pergunta, variante, temperatura, top_p, max_tokens, penalidade):
    if not pergunta.strip():
        yield "Digite uma pergunta.", ""
        return
    if tem_adapter and variante.startswith("Base"):
        ctx = modelo.disable_adapter()
    else:
        ctx = contextlib.nullcontext()
    t0 = time.time()
    texto, n = "", 0
    with ctx:
        for texto, n in gerar_stream(modelo, tok, device, pergunta, temperatura=temperatura,
                                     top_p=top_p, max_novos_tokens=int(max_tokens),
                                     penalidade=penalidade):
            yield texto, f"{n} tokens | {n / max(time.time() - t0, 1e-6):.1f} tokens/s"
    seg = time.time() - t0
    yield texto, f"{n} tokens em {seg:.1f}s | {n / max(seg, 1e-6):.1f} tokens/s | {nome_gpu} ({backend})"


with gr.Blocks(title="Tutor de Python (LLM local)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Tutor de Python para iniciantes\nLLM local ajustada com LoRA, rodando na GPU.")
    gr.Markdown(f"`{info_gpu}`")
    with gr.Row():
        with gr.Column(scale=3):
            pergunta = gr.Textbox(label="Sua pergunta", lines=4,
                                  placeholder="Ex.: Como funciona o laco for em Python?")
            botao = gr.Button("Enviar", variant="primary")
            resposta = gr.Textbox(label="Resposta", lines=14, show_copy_button=True)
            stats = gr.Textbox(label="Desempenho", interactive=False)
        with gr.Column(scale=1):
            variante = gr.Radio(OPCOES, value=OPCOES[0], label="Modelo")
            temperatura = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="Temperatura (0 = deterministico)")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
            max_tokens = gr.Slider(32, 400, value=200, step=8, label="Maximo de tokens")
            penalidade = gr.Slider(1.0, 1.5, value=1.1, step=0.05, label="Penalidade de repeticao")
    entradas = [pergunta, variante, temperatura, top_p, max_tokens, penalidade]
    botao.click(responder, entradas, [resposta, stats])
    pergunta.submit(responder, entradas, [resposta, stats])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=args.porta)
