# Use a pipeline as a high-level helper
from transformers import pipeline

# Load the model on GPU #0 memory
# pipe = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca", device="cuda:0")

# Load the model on CPU memory
# model = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca", device="cpu").model
# Cast to half precision.
model = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca", device="cpu").model.half().half()
# Save locally.
model.save_pretrained(r"D:\model_quarter.mdl")

# pipe = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca")
a = 1