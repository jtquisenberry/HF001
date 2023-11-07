# Use a pipeline as a high-level helper
import torch
from transformers import pipeline

torch.device(0)
device = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(device=device)
print("Current Device")
print(properties)
print()

# Load the model on GPU #0 memory
# pipe = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca", device="cuda:0")

# Load the model on CPU memory
# model = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca", device="cpu").model
# Cast to half precision.
# model = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca", device="cpu").model.half().half()
# model = pipeline("text-generation", model="Open-Orca/LlongOrca-7B-16k", device="cpu", torch_dtype=torch.float16).model
model = pipeline("text-generation", model="Open-Orca/LlongOrca-7B-16k", device=0, torch_dtype=torch.float16).model
model = pipeline("text-generation", model="Open-Orca/LlongOrca-7B-16k", device=0).model
# Save locally.
# model.save_pretrained(r"D:\model_f16.mdl")

# pipe = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca")
a = 1