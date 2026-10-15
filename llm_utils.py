"""Funcoes compartilhadas: carregar modelo/tokenizer e gerar texto.

A geracao usa um laco proprio: o modelo roda na GPU e so o sorteio do proximo
token (softmax/top-p/multinomial) roda na CPU. Isso evita operacoes que o
DirectML ainda nao suporta bem e permite transmitir o texto aos poucos no frontend.
"""
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO_PADRAO = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "Voce e um tutor de Python para iniciantes. Responda sempre em portugues, "
    "de forma curta e clara, e mostre um exemplo de codigo quando ajudar."
)


def montar_mensagens(pergunta):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pergunta},
    ]


def carregar_tokenizer(nome=MODELO_PADRAO):
    tok = AutoTokenizer.from_pretrained(nome)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def carregar_modelo_base(nome=MODELO_PADRAO):
    # float32: o DirectML ainda e mais estavel assim que em fp16/bf16
    # eager: atencao com operacoes basicas (matmul/softmax), mais segura no DirectML que o sdpa
    return AutoModelForCausalLM.from_pretrained(
        nome, torch_dtype=torch.float32, attn_implementation="eager"
    )


def mascara_causal_4d(mask2d, q_len, kv_len, device, dtype=torch.float32):
    """Mascara aditiva (B, 1, q_len, kv_len): 0 = pode olhar, valor minimo = bloqueado.

    E construida na CPU e depois enviada a GPU, porque o masked_fill que o transformers
    usa internamente falha no DirectML ("value cannot be converted to type uint8_t").
    Quando recebe uma mascara 4D pronta, o transformers nao roda esse trecho.
    mask2d: tensor (B, kv_len) com 1 = token real e 0 = padding (ou None).
    """
    minimo = torch.finfo(dtype).min
    pos_q = torch.arange(kv_len - q_len, kv_len).unsqueeze(1)  # posicao absoluta de cada query
    pos_k = torch.arange(kv_len).unsqueeze(0)
    bloqueado = (pos_k > pos_q)[None, None]  # (1, 1, q, kv) - nao olhar o futuro
    if mask2d is not None:
        bloqueado = bloqueado | (mask2d.cpu() == 0)[:, None, None, :]  # nao olhar padding
    mascara = torch.zeros(bloqueado.shape, dtype=dtype).masked_fill(bloqueado, minimo)
    return mascara.to(device)


def carregar_para_inferencia(nome, adapter, device):
    """Carrega o modelo base e, se existir, o adaptador LoRA treinado.
    Retorna (modelo, tem_adapter)."""
    from peft import PeftModel

    modelo = carregar_modelo_base(nome)
    tem_adapter = bool(adapter) and os.path.isdir(adapter)
    if tem_adapter:
        modelo = PeftModel.from_pretrained(modelo, adapter)
    modelo.to(device)
    modelo.eval()
    return modelo, tem_adapter


def _amostrar(logits, temperatura, top_p, gerados, penalidade):
    """Escolhe o proximo token (tudo na CPU)."""
    if penalidade != 1.0 and gerados:
        idx = torch.tensor(sorted(set(gerados)))
        v = logits[idx]
        logits[idx] = torch.where(v > 0, v / penalidade, v * penalidade)
    if temperatura <= 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits / temperatura, dim=-1)
    p_ord, i_ord = torch.sort(probs, descending=True)
    acum = torch.cumsum(p_ord, dim=0)
    p_ord[(acum - p_ord) > top_p] = 0.0
    p_ord = p_ord / p_ord.sum()
    return int(i_ord[torch.multinomial(p_ord, 1)])


@torch.no_grad()
def gerar_stream(modelo, tok, device, pergunta, temperatura=0.7, top_p=0.9,
                 max_novos_tokens=200, penalidade=1.1):
    """Gera a resposta token a token. A cada passo entrega (texto_ate_agora, n_tokens)."""
    texto = tok.apply_chat_template(
        montar_mensagens(pergunta), tokenize=False, add_generation_prompt=True
    )
    entrada = tok(texto, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    paradas = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id:
            paradas.add(tid)

    gerados, passado, atual, comprimento = [], None, entrada, 0
    for _ in range(max_novos_tokens):
        q = atual.shape[1]
        mascara = mascara_causal_4d(None, q, comprimento + q, device)
        saida = modelo(input_ids=atual, attention_mask=mascara,
                       past_key_values=passado, use_cache=True)
        comprimento += q
        passado = saida.past_key_values
        logits = saida.logits[0, -1, :].float().cpu()
        prox = _amostrar(logits, temperatura, top_p, gerados, penalidade)
        if prox in paradas:
            break
        gerados.append(prox)
        atual = torch.tensor([[prox]], device=device)
        yield tok.decode(gerados, skip_special_tokens=True), len(gerados)
    yield tok.decode(gerados, skip_special_tokens=True), len(gerados)


def gerar(modelo, tok, device, pergunta, **kw):
    """Versao simples: devolve (texto, n_tokens, segundos)."""
    t0 = time.time()
    texto, n = "", 0
    for texto, n in gerar_stream(modelo, tok, device, pergunta, **kw):
        pass
    return texto, n, time.time() - t0
