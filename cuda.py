import torch

# checks if cuda is available on this device and what gpu there is
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
