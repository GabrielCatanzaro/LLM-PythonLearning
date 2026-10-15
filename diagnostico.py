"""Diagnostico: compara a GPU (DirectML) com a CPU para achar operacoes que dao resultado errado.

    python diagnostico.py

Parte A: testa operacoes isoladas usadas pelo modelo (GPU x CPU).
Parte B: roda o modelo base nos dois e compara a loss e cada camada.
Leva ~1 a 2 minutos. Copie a saida inteira e envie.
"""
import torch
import torch.nn.functional as F

from gpu_utils import descrever, get_device
from llm_utils import carregar_modelo_base, carregar_tokenizer, mascara_causal_4d, montar_mensagens


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).abs().max() / (b.abs().max() + 1e-8)).item()


def parte_a(dev):
    print("\n=== PARTE A: operacoes isoladas (GPU x CPU) ===")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 16, 64, generator=g)
    w = torch.randn(64, 64, generator=g)
    q = torch.randn(2, 14, 16, 64, generator=g)
    kv = torch.randn(2, 2, 16, 64, generator=g)
    mask = mascara_causal_4d(None, 16, 16, "cpu")
    ids = torch.randint(0, 1000, (2, 16), generator=g)
    emb = torch.randn(1000, 64, generator=g)
    labels = ids.clone()
    labels[:, :5] = -100
    logits = torch.randn(2, 16, 1000, generator=g)
    freq = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))

    casos = {
        "matmul": lambda T: T(x) @ T(w),
        "rmsnorm (pow/mean/rsqrt)": lambda T: T(x) * torch.rsqrt(T(x).pow(2).mean(-1, keepdim=True) + 1e-6),
        "softmax com mascara aditiva": lambda T: torch.softmax(T(q) @ T(q).transpose(-1, -2) / 8 + T(mask), dim=-1),
        "repeat_kv (expand+reshape)": lambda T: T(kv)[:, :, None, :, :].expand(2, 2, 7, 16, 64).reshape(2, 14, 16, 64),
        "embedding": lambda T: F.embedding(T(ids), T(emb)),
        "silu * matmul (MLP)": lambda T: F.silu(T(x)) * (T(x) @ T(w)),
        "rotary (cat/neg/slice)": lambda T: torch.cat((-T(x)[..., 32:], T(x)[..., :32]), dim=-1),
        "cos/sin": lambda T: torch.cos(T(x)) + torch.sin(T(x)),
        "inv_freq * posicao": lambda T: T(torch.arange(16).float())[:, None] * T(freq)[None, :],
        "transpose+reshape": lambda T: T(q).transpose(1, 2).reshape(2, 16, -1),
        "cross_entropy ignore_index": lambda T: F.cross_entropy(
            T(logits).reshape(-1, 1000), T(labels).reshape(-1), ignore_index=-100),
    }
    ruins = []
    for nome, f in casos.items():
        try:
            err = rel(f(lambda t: t.to(dev)), f(lambda t: t))
            status = "OK  " if err < 1e-3 else "ERRO"
            if status == "ERRO":
                ruins.append(nome)
            print(f"{status} {nome}: erro relativo {err:.2e}")
        except Exception as e:
            ruins.append(nome)
            print(f"FALHOU {nome}: {str(e)[:120]}")
    return ruins


def parte_b(dev):
    print("\n=== PARTE B: modelo completo (GPU x CPU) ===")
    tok = carregar_tokenizer()
    modelo = carregar_modelo_base()
    modelo.eval()
    texto = tok.apply_chat_template(
        montar_mensagens("Como imprimir uma mensagem na tela?"), tokenize=False, add_generation_prompt=True
    ) + "Use a funcao print(). Ela mostra na tela o texto que voce colocar entre parenteses." + tok.eos_token
    ids = tok(texto, return_tensors="pt", add_special_tokens=False).input_ids
    T = ids.shape[1]
    mask = mascara_causal_4d(torch.ones(1, T, dtype=torch.long), T, T, "cpu")

    with torch.no_grad():
        ref = modelo(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    loss_cpu = F.cross_entropy(ref.logits[0, :-1].float(), ids[0, 1:]).item()

    modelo.to(dev)
    with torch.no_grad():
        out = modelo(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True)
    loss_gpu = F.cross_entropy(out.logits[0, :-1].float().cpu(), ids[0, 1:]).item()

    print(f"Loss do modelo base na CPU: {loss_cpu:.3f}  (esperado: algo entre 1 e 4)")
    print(f"Loss do modelo base na GPU: {loss_gpu:.3f}")
    primeiro = None
    for i, (a, b) in enumerate(zip(out.hidden_states, ref.hidden_states)):
        err = rel(a, b)
        marca = "" if err < 1e-2 else "  <-- DIVERGE"
        if marca and primeiro is None:
            primeiro = i
        print(f"saida da camada {i:2d}: erro relativo {err:.2e}{marca}")
    if abs(loss_cpu - loss_gpu) < 0.1:
        print("\nRESULTADO: a GPU calcula igual a CPU. O problema esta no treino/loss, nao no modelo.")
    else:
        onde = "ja nos embeddings" if primeiro == 0 else f"a partir da camada {primeiro}"
        print(f"\nRESULTADO: a GPU calcula DIFERENTE da CPU ({onde}).")


def main():
    dev, backend, nome = get_device()
    print(descrever(dev, backend, nome))
    ruins = parte_a(dev)
    if ruins:
        print("\nOperacoes com problema:", ruins)
    parte_b(dev)


if __name__ == "__main__":
    main()
