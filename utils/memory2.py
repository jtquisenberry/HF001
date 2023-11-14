import pynvml
import psutil
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(h)
print(f'total    : {round(info.total/1024.0/1024.0/1024.0, 2):.2f}')
print(f'free     : {round(info.free/1024.0/1024.0/1024.0, 2)}')
print(f'used     : {round(info.used/1024.0/1024.0/1024.0, 2):.2f}')
