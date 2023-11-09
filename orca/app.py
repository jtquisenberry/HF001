# Use a pipeline as a high-level helper
import torch
from transformers import pipeline
from threading import Thread
import time
import pynvml
import psutil
import os
import sys
from datetime import datetime


import logging
log_path = './my_log.log'

import logging
#logging.basicConfig(filemode='a',
#                    level=logging.INFO)

log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger("main")
file_handler = logging.FileHandler(filename=log_path, mode='a')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)



"""
logging.basicConfig(filename=log_name,
                    filemode='a',
                    format='%(asctime)s, %(name)s %(levelname)s %(message)s',
                    level=logging.INFO)
logger = logging.getLogger("main")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
logger.addHandler(logging.StreamHandler(sys.stdout))
"""
#logger.info("AAAAAAAAAAAAAa")
#a = 1


class ResourceLogger(Thread):
    def __init__(self):
        super().__init__()
        self.terminate = False
        self.times = []
        self.gpups = []
        self.memps = []

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
            #print(f"GPU VMEM {gpu_mem_used:.2f} of {gpu_mem_total:.2f} ({gpu_mem_p * 100:.2f}%), "
            #      f"Sys RAM {sys_mem_used:.2f} of {sys_mem_total:.2f} ({sys_mem_p * 100:.2f}%)")
            logger.info(f"GPU VMEM {gpu_mem_used:.2f} of {gpu_mem_total:.2f} ({gpu_mem_p * 100:.2f}%), "
                        f"Sys RAM {sys_mem_used:.2f} of {sys_mem_total:.2f} ({sys_mem_p * 100:.2f}%)")
            dt = str(datetime.now())
            self.times.append(dt)
            self.gpups.append(gpu_mem_p)
            self.memps.append(sys_mem_p)
            time.sleep(2)

    def stop(self):
        self.terminate = True


def main():
    model = pipeline("text-generation", model="Open-Orca/LlongOrca-7B-16k", device=0, torch_dtype=torch.float16).model
    filename = "orca.mdl"
    fq_filename = f"/mnt/d/{filename}"
    model.save_pretrained(fq_filename)
    file_size = os.path.getsize(fq_filename)
    logger.info(f"Filename: {filename}, File size {file_size}")
    print("DONE")


if __name__ == "__main__":
    resource_logger = ResourceLogger()
    resource_logger.start()
    main()
    logger.warning(f"Times: {resource_logger.times}")
    logger.warning(f"GPU%s: {resource_logger.gpups}")
    logger.warning(f"MEM%s: {resource_logger.memps}")
    resource_logger.stop()
    a = 1
