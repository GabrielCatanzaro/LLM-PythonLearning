"""Selecao e informacoes da GPU.

Placas AMD (ex.: RX 6600) nao usam CUDA. No Windows usamos o backend DirectML
(pacote torch-directml). Se existir uma GPU NVIDIA com CUDA, ela tambem e aceita,
o que permite que colegas do grupo com NVIDIA rodem o mesmo codigo.
"""
import torch


def listar_gpus_directml():
    """Retorna lista de (indice, nome) das GPUs vistas pelo DirectML."""
    try:
        import torch_directml
    except Exception:
        return []
    try:
        return [(i, torch_directml.device_name(i)) for i in range(torch_directml.device_count())]
    except Exception:
        return []


def get_device(indice=None, permitir_cpu=False):
    """Escolhe o dispositivo. Retorna (device, backend, nome).

    indice: indice da GPU DirectML (use verificar_gpu.py para ver a lista).
            Se None, tenta achar uma placa 'RX' (sua RX 6600) e, se nao achar,
            usa a GPU padrao.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", torch.cuda.get_device_name(0)

    gpus = listar_gpus_directml()
    if gpus:
        import torch_directml

        if indice is None:
            indice = next((i for i, n in gpus if "rx" in n.lower()), None)
            if indice is None:
                indice = torch_directml.default_device()
        nome = dict(gpus).get(indice, f"GPU {indice}")
        return torch_directml.device(indice), "directml", nome

    if permitir_cpu:
        return torch.device("cpu"), "cpu", "CPU"

    raise RuntimeError(
        "Nenhuma GPU encontrada. Rode `python verificar_gpu.py` para diagnosticar. "
        "Para testar sem GPU (nao vale para a entrega) use --permitir_cpu."
    )


def descrever(device, backend, nome):
    return f"[GPU] backend={backend} | dispositivo={device} | nome={nome} | torch={torch.__version__}"
