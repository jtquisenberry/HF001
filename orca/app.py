# Use a pipeline as a high-level helper
import torch
from transformers import pipeline
from threading import Thread
import time
import pynvml
import psutil


class ResourceLogger(Thread):
    def __init__(self):
        super().__init__()
        self.terminate = False

    def run(self):
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not self.terminate:
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpu_mem_used = round(info.used / 1024.0 / 1024.0 / 1024.0, 2)
            gpu_mem_total = round(info.total / 1024.0 / 1024.0 / 1024.0, 2)
            gpu_mem_p = round(gpu_mem_used / gpu_mem_total, 2)
            sys_mem_used = round(psutil.virtual_memory().used / 1024.0 / 1024.0 / 1024.0, 2)
            sys_mem_total = round(psutil.virtual_memory().total / 1024.0 / 1024.0 / 1024.0, 2)
            sys_mem_p = round(sys_mem_used / sys_mem_total, 2)
            print(f"GPU VMEM {gpu_mem_used:.2f} of {gpu_mem_total:.2f} ({gpu_mem_p * 100:.2f}%), "
                  f"Sys RAM {sys_mem_used:.2f} of {sys_mem_total:.2f} ({sys_mem_p * 100:.2f}%)")
            time.sleep(2)

    def stop(self):
        self.terminate = True


def main():
    model = pipeline("text-generation", model="Open-Orca/LlongOrca-7B-16k", device=0, torch_dtype=torch.float16).model
    model.save_pretrained(r"D:\model_f16.mdl")


if __name__ == "__main__":
    resource_logger = ResourceLogger()
    resource_logger.start()
    main()
    resource_logger.stop()
    a = 1
