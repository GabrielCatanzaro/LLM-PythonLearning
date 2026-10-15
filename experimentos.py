"""Roda varios treinos com parametros diferentes e compara os resultados.

    python experimentos.py

Saidas: resultados/experimentos.csv e resultados/comparacao_loss.png
Edite a lista CONFIGS para testar outros valores.
"""
import argparse
import csv
import gc
import os

from train import criar_parser, treinar

CONFIGS = [
    {"nome": "base_lr3e-4_r8_e5", "lr": 3e-4, "rank": 8, "alpha": 16, "epocas": 5},
    {"nome": "lr1e-4_r8_e5", "lr": 1e-4, "rank": 8, "alpha": 16, "epocas": 5},
    {"nome": "lr1e-3_r8_e5", "lr": 1e-3, "rank": 8, "alpha": 16, "epocas": 5},
    {"nome": "lr3e-4_r16_e5", "lr": 3e-4, "rank": 16, "alpha": 32, "epocas": 5},
    {"nome": "lr3e-4_r8_e10", "lr": 3e-4, "rank": 8, "alpha": 16, "epocas": 10},
]

CAMPOS = ["nome", "lr", "rank", "alpha", "epocas", "loss_treino", "loss_val", "tempo_s", "gpu", "erro"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--permitir_cpu", action="store_true")
    extra = ap.parse_args()

    padrao = vars(criar_parser().parse_args([]))
    linhas = []
    for cfg in CONFIGS:
        print("\n" + "=" * 60 + f"\nEXPERIMENTO: {cfg['nome']}\n" + "=" * 60)
        args = argparse.Namespace(**{**padrao, **cfg, "gpu": extra.gpu, "permitir_cpu": extra.permitir_cpu})
        try:
            r = treinar(args)
            linhas.append({**{k: r.get(k) for k in CAMPOS}, "erro": ""})
        except Exception as e:  # um experimento com erro nao derruba os outros
            print(f"[ERRO] {cfg['nome']}: {e}")
            linhas.append({"nome": cfg["nome"], "lr": cfg["lr"], "rank": cfg["rank"],
                           "alpha": cfg["alpha"], "epocas": cfg["epocas"], "erro": str(e)[:200]})
        gc.collect()

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/experimentos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(linhas)
    print("\nResumo salvo em resultados/experimentos.csv")
    for l in linhas:
        print(l)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for cfg in CONFIGS:
            caminho = f"resultados/loss_{cfg['nome']}.csv"
            if not os.path.exists(caminho):
                continue
            with open(caminho, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            plt.plot([int(r["passo"]) for r in rows], [float(r["loss"]) for r in rows], label=cfg["nome"])
        plt.xlabel("passo de otimizacao")
        plt.ylabel("loss de treino")
        plt.title("Comparacao de parametros")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("resultados/comparacao_loss.png", dpi=150)
        print("Grafico salvo em resultados/comparacao_loss.png")
    except Exception as e:
        print(f"(grafico nao gerado: {e})")


if __name__ == "__main__":
    main()
