"""Fine-tuning LoRA do Qwen2.5-0.5B-Instruct como tutor de Python.

Uso basico:
    python train.py
Exemplo com outros parametros:
    python train.py --lr 1e-4 --rank 16 --alpha 32 --epocas 8 --nome teste_r16
"""
import argparse
import csv
import gc
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from gpu_utils import descrever, get_device
from llm_utils import (MODELO_PADRAO, carregar_modelo_base, carregar_tokenizer,
                       mascara_causal_4d, montar_mensagens)


def criar_parser():
    p = argparse.ArgumentParser(description="Fine-tuning LoRA (GPU via DirectML/CUDA)")
    p.add_argument("--modelo", default=MODELO_PADRAO)
    p.add_argument("--dados", default="data/dataset.jsonl")
    p.add_argument("--epocas", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--acum", type=int, default=2, help="passos de acumulo de gradiente")
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--nome", default=None, help="nome da execucao (pasta em modelos/)")
    p.add_argument("--gpu", type=int, default=None, help="indice da GPU DirectML")
    p.add_argument("--permitir_cpu", action="store_true", help="so para teste; a entrega exige GPU")
    p.add_argument("--max_passos", type=int, default=None, help="limita passos (teste rapido)")
    return p


def ler_dataset(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def tokenizar(tok, pares, max_len):
    """Loss so nos tokens da resposta (o prompt recebe label -100)."""
    exemplos = []
    for p in pares:
        prompt = tok.apply_chat_template(
            montar_mensagens(p["pergunta"]), tokenize=False, add_generation_prompt=True
        )
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        r_ids = tok(p["resposta"] + tok.eos_token, add_special_tokens=False).input_ids
        ids = (p_ids + r_ids)[:max_len]
        labels = ([-100] * len(p_ids) + r_ids)[:max_len]
        exemplos.append((ids, labels))
    return exemplos


def montar_batch(lote, pad_id, device):
    tam = max(len(ids) for ids, _ in lote)
    ids_b, mask_b, lab_b = [], [], []
    for ids, labels in lote:
        pad = tam - len(ids)
        ids_b.append(ids + [pad_id] * pad)
        mask_b.append([1] * len(ids) + [0] * pad)
        lab_b.append(labels + [-100] * pad)
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return t(ids_b), t(mask_b), t(lab_b)


ESTADO = {"loss_cpu": False}


def _entropia(logits, alvo):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), alvo.reshape(-1), ignore_index=-100
    )


def calcular_loss(modelo, ids, mask, labels, loss_cpu=None):
    # Mascara 4D pronta (feita na CPU): evita o masked_fill que o DirectML nao suporta.
    mascara = mascara_causal_4d(mask, ids.shape[1], ids.shape[1], ids.device)
    logits = modelo(input_ids=ids, attention_mask=mascara).logits[:, :-1, :]
    alvo = labels[:, 1:]
    # F.cross_entropy com ignore_index da resultado ERRADO no DirectML (confirmado pelo
    # diagnostico.py). Por isso a loss e sempre calculada na CPU; o gradiente volta
    # normalmente para a GPU, e o resto do modelo continua rodando na GPU.
    return _entropia(logits.cpu(), alvo.cpu())


@torch.no_grad()
def loss_validacao(modelo, dados, pad_id, device, batch, loss_cpu):
    if not dados:
        return float("nan")
    modelo.eval()
    total, n = 0.0, 0
    for i in range(0, len(dados), batch):
        ids, mask, lab = montar_batch(dados[i:i + batch], pad_id, device)
        total += calcular_loss(modelo, ids, mask, lab, loss_cpu).item() * len(ids)
        n += len(ids)
    modelo.train()
    return total / n


def treinar(args):
    random.seed(args.semente)
    torch.manual_seed(args.semente)

    device, backend, nome_gpu = get_device(args.gpu, args.permitir_cpu)
    print(descrever(device, backend, nome_gpu))

    nome = args.nome or f"lr{args.lr:g}_r{args.rank}_e{args.epocas}"
    saida = os.path.join("modelos", nome)
    os.makedirs(saida, exist_ok=True)
    os.makedirs("resultados", exist_ok=True)

    tok = carregar_tokenizer(args.modelo)
    modelo = carregar_modelo_base(args.modelo)
    cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(modelo, cfg)
    modelo.print_trainable_parameters()
    modelo.to(device)
    modelo.train()

    pares = ler_dataset(args.dados)
    random.Random(args.semente).shuffle(pares)
    n_val = max(1, int(len(pares) * args.val_frac))
    exemplos = tokenizar(tok, pares, args.max_len)
    val, treino = exemplos[:n_val], exemplos[n_val:]
    print(f"Exemplos: {len(treino)} treino | {len(val)} validacao")

    params = [p for p in modelo.parameters() if p.requires_grad]
    otim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    passos_por_epoca = max(1, (len(treino) + args.batch * args.acum - 1) // (args.batch * args.acum))
    total_passos = passos_por_epoca * args.epocas
    aquecimento = max(1, int(0.1 * total_passos))
    sched = torch.optim.lr_scheduler.LambdaLR(
        otim,
        lambda s: (s + 1) / aquecimento if s < aquecimento
        else max(0.0, (total_passos - s) / max(1, total_passos - aquecimento)),
    )

    loss_cpu = False
    historico, passo, t0 = [], 0, time.time()
    rng = random.Random(args.semente)
    for epoca in range(1, args.epocas + 1):
        rng.shuffle(treino)
        losses_epoca = []
        otim.zero_grad()
        lotes = [treino[i:i + args.batch] for i in range(0, len(treino), args.batch)]
        for k, lote in enumerate(lotes, 1):
            ids, mask, lab = montar_batch(lote, tok.pad_token_id, device)
            loss = calcular_loss(modelo, ids, mask, lab)
            if passo == 0 and k == 1:
                print(f"Loss inicial: {loss.item():.3f} (esperado: entre 1 e 4; "
                      f"se for maior que 8, ha algo errado)")
            (loss / args.acum).backward()
            losses_epoca.append(loss.item())

            if k % args.acum == 0 or k == len(lotes):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                otim.step()
                sched.step()
                otim.zero_grad()
                passo += 1
                lr_atual = sched.get_last_lr()[0]
                media = sum(losses_epoca[-args.acum:]) / len(losses_epoca[-args.acum:])
                historico.append((passo, epoca, media, lr_atual))
                print(f"epoca {epoca}/{args.epocas} passo {passo}/{total_passos} "
                      f"loss {media:.4f} lr {lr_atual:.2e} | {time.time() - t0:.0f}s")
                if args.max_passos and passo >= args.max_passos:
                    break
        if args.max_passos and passo >= args.max_passos:
            break

    tempo = time.time() - t0
    loss_val = loss_validacao(modelo, val, tok.pad_token_id, device, args.batch, loss_cpu)
    loss_treino = sum(h[2] for h in historico[-passos_por_epoca:]) / max(1, len(historico[-passos_por_epoca:]))
    print(f"Fim: loss treino {loss_treino:.4f} | loss validacao {loss_val:.4f} | {tempo:.0f}s")

    modelo.to("cpu")  # salvar a partir da CPU evita problemas com tensores na GPU
    modelo.save_pretrained(saida)
    tok.save_pretrained(saida)

    resultado = {
        "nome": nome, "modelo": args.modelo, "backend": backend, "gpu": nome_gpu,
        "epocas": args.epocas, "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
        "dropout": args.dropout, "batch": args.batch, "acum": args.acum,
        "n_treino": len(treino), "n_val": len(val), "passos": passo,
        "loss_treino": round(loss_treino, 4), "loss_val": round(loss_val, 4),
        "tempo_s": round(tempo, 1), "saida": saida,
    }
    with open(os.path.join("resultados", f"treino_{nome}.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    with open(os.path.join("resultados", f"loss_{nome}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["passo", "epoca", "loss", "lr"])
        w.writerows(historico)

    del modelo, otim
    gc.collect()
    return resultado


if __name__ == "__main__":
    treinar(criar_parser().parse_args())
