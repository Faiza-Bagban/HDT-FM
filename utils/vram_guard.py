# utils/vram_guard.py
import torch

def vram_check(threshold=0.88):
    if not torch.cuda.is_available():
        return
    used = torch.cuda.memory_reserved(0)
    total = torch.cuda.get_device_properties(0).total_memory
    ratio = used / total
    if ratio > threshold:
        raise RuntimeError(f"VRAM {ratio:.1%} > {threshold:.0%} limit. Reduce batch or enable grad ckpt.")
    print(f"VRAM: {used/1e9:.2f} GB / {total/1e9:.2f} GB ({ratio:.1%})")