"""Mostra as informacoes da GPU e confirma que ela funciona.

Rode no terminal do VSCode e tire o print para a entrega:
    python verificar_gpu.py
"""
import platform
import sys
import time

import torch

from gpu_utils import descrever, get_device, listar_gpus_directml


def main():
    print("=" * 60)
    print("VERIFICACAO DE GPU")
    print("=" * 60)
    print(f"Python : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponivel (NVIDIA): {torch.cuda.is_available()}")

    gpus = listar_gpus_directml()
    print("GPUs vistas pelo DirectML:")
    if not gpus:
        print("  (nenhuma) -> verifique `pip install torch-directml` e o driver AMD")
    for i, nome in gpus:
        print(f"  indice {i}: {nome}")

    device, backend, nome = get_device(permitir_cpu=True)
    print(descrever(device, backend, nome))
    if backend == "cpu":
        print("ATENCAO: rodando em CPU. A atividade exige GPU.")
        return

    # Teste 1: multiplicacao de matrizes (GPU vs CPU)
    n = 2048
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    t0 = time.time()
    for _ in range(3):
        (a @ b).sum().item()
    t_cpu = (time.time() - t0) / 3

    ag, bg = a.to(device), b.to(device)
    (ag @ bg).sum().item()  # aquecimento
    t0 = time.time()
    for _ in range(10):
        (ag @ bg).sum().item()
    t_gpu = (time.time() - t0) / 10
    print(f"matmul {n}x{n}: CPU {t_cpu*1000:.1f} ms | GPU {t_gpu*1000:.1f} ms | ganho ~{t_cpu/t_gpu:.1f}x")

    # Teste 2: forward + backward (treino precisa de autograd)
    modelo = torch.nn.Linear(512, 512).to(device)
    x = torch.randn(32, 512, device=device)
    perda = modelo(x).pow(2).mean()
    perda.backward()
    print(f"forward/backward na GPU OK (perda={perda.item():.4f})")
    print("GPU OK - pode treinar.")


if __name__ == "__main__":
    main()
