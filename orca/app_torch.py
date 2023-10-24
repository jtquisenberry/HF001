import torch

# GPUs
print(f"Number of GPUs: {torch.cuda.device_count()}")
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
for gpu in available_gpus:
    print("----------------------------")
    print(f"GPU Index: {gpu.idx}")
    properties = torch.cuda.get_device_properties(gpu.idx)
    print(properties)
    print("----------------------------")
    print()

# Current Device
device = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(device=device)
print("Current Device")
print(properties)
print()

# Set Device
# device = "cpu"
device_idx = "cuda:0"
torch.device(device_idx)
device = torch.cuda.current_device()

t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved

debug_breakpoint = 1