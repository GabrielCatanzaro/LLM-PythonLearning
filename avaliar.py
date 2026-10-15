"""Avalia as respostas: modelo base x modelo ajustado (LoRA).

    python avaliar.py --adapter modelos/base_lr3e-4_r8_e5
    python avaliar.py --adapter modelos/base_lr3e-4_r8_e5 --rapido      (so 5 perguntas)

Metricas:
  - cobertura: fracao das palavras-chave esperadas que aparecem na resposta (0 a 1)
  - nota_manual: coluna em branco no CSV para voces preencherem (1 a 5)
Saidas: resultados/avaliacao.csv e resultados/avaliacao.md
"""
import argparse
import csv
import json
import os
from collections import defaultdict

from gpu_utils import descrever, get_device
from llm_utils import MODELO_PADRAO, carregar_para_inferencia, carregar_tokenizer, gerar


def cobertura(resposta, palavras):
    r = resposta.lower()
    achadas = [p for p in palavras if p.lower() in r]
    return len(achadas) / len(palavras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=MODELO_PADRAO)
    ap.add_argument("--adapter", required=True, help="pasta do adaptador LoRA treinado")
    ap.add_argument("--testes", default="data/perguntas_teste.json")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.7],
                    help="temperaturas a testar (0 = deterministico)")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=150)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--permitir_cpu", action="store_true")
    args = ap.parse_args()

    device, backend, nome_gpu = get_device(args.gpu, args.permitir_cpu)
    print(descrever(device, backend, nome_gpu))

    with open(args.testes, encoding="utf-8") as f:
        testes = json.load(f)
    if args.rapido:
        testes = testes[:5]

    tok = carregar_tokenizer(args.modelo)
    modelo, tem = carregar_para_inferencia(args.modelo, args.adapter, device)
    if not tem:
        raise SystemExit(f"Adaptador nao encontrado em {args.adapter}. Rode train.py primeiro.")

    linhas = []
    for temp in args.temps:
        for variante in ("base", "ajustado"):
            print(f"\n--- {variante} | temperatura {temp} ---")
            for t in testes:
                kw = dict(temperatura=temp, top_p=args.top_p, max_novos_tokens=args.max_tokens)
                if variante == "base":
                    with modelo.disable_adapter():
                        resp, n, seg = gerar(modelo, tok, device, t["pergunta"], **kw)
                else:
                    resp, n, seg = gerar(modelo, tok, device, t["pergunta"], **kw)
                cob = cobertura(resp, t["palavras_chave"])
                linhas.append({
                    "modelo": variante, "temperatura": temp, "pergunta": t["pergunta"],
                    "resposta": resp.replace("\n", " \\n "), "cobertura": round(cob, 2),
                    "tokens": n, "tok_por_s": round(n / seg, 1) if seg else 0, "nota_manual": "",
                })
                print(f"[{cob:.2f}] {t['pergunta']}  ({n} tok, {n / seg:.1f} tok/s)")

    os.makedirs("resultados", exist_ok=True)
    campos = list(linhas[0].keys())
    with open("resultados/avaliacao.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)

    resumo = defaultdict(list)
    for l in linhas:
        resumo[(l["modelo"], l["temperatura"])].append((l["cobertura"], l["tok_por_s"]))

    with open("resultados/avaliacao.md", "w", encoding="utf-8") as f:
        f.write(f"# Avaliacao ({backend}: {nome_gpu})\n\n")
        f.write("| modelo | temperatura | cobertura media | tokens/s medio |\n|---|---|---|---|\n")
        for (m, temp), v in sorted(resumo.items(), key=lambda x: (x[0][1], x[0][0])):
            cob = sum(a for a, _ in v) / len(v)
            vel = sum(b for _, b in v) / len(v)
            f.write(f"| {m} | {temp} | {cob:.2f} | {vel:.1f} |\n")
        f.write("\n## Respostas\n")
        for l in linhas:
            f.write(f"\n**[{l['modelo']} | T={l['temperatura']}] {l['pergunta']}** (cobertura {l['cobertura']})\n\n")
            f.write(l["resposta"].replace(" \\n ", "\n") + "\n")

    print("\nResumo:")
    for (m, temp), v in sorted(resumo.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"  {m:9s} T={temp}: cobertura media {sum(a for a, _ in v) / len(v):.2f}")
    print("Arquivos: resultados/avaliacao.csv e resultados/avaliacao.md")


if __name__ == "__main__":
    main()
