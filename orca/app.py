# Use a pipeline as a high-level helper
import torch
from transformers import pipeline
from multiprocessing import Process
import time
import pynvml
import psutil


def log_memory():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    for i in range(1000):
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        #print(f'total    : {round(info.total / 1024.0 / 1024.0 / 1024.0, 2):.2f}')
        #print(f'free     : {round(info.free / 1024.0 / 1024.0 / 1024.0, 2)}')
        print(f'GPU VMEM Used  : {round(info.used / 1024.0 / 1024.0 / 1024.0, 2):.2f}')
        print(f'System RAM     : {round(psutil.virtual_memory().used / 1024.0 / 1024.0 / 1024.0, 2):.2f}')
        time.sleep(2)


def main():
    # Load the model on CPU memory
    # model = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca", device="cpu").model
    # Cast to half precision.
    # model = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca", device="cpu").model.half().half()
    #model = pipeline("text-generation", model="Open-Orca/LlongOrca-7B-16k", device="cpu", torch_dtype=torch.float16).model
    model = pipeline("text-generation", model="Open-Orca/LlongOrca-7B-16k", device=0, torch_dtype=torch.float16).model
    # Save locally.
    # model.save_pretrained(r"D:\model_f16.mdl")

    # pipe = pipeline("text-generation", model="Open-Orca/Mistral-7B-OpenOrca")
    time.sleep(120)
    a = 1

if __name__ == "__main__":
    p = Process(target=log_memory)
    p.start()
    main()
