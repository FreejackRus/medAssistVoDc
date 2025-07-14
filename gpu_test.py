import torch

print(f"Доступен CUDA: {torch.cuda.is_available()}")
print(f"Количество GPU: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Название GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")