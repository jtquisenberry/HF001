import torch
device = torch.cuda.current_device()



m = torch.cuda.mem_get_info(device)
free = m[0]/1024.0/1024.0
total = m[1]/1024.0/1024.0
used = (m[1] - m[0])/1024.0/1024.0

print(f"Total: {total}, Used: {used}, Free {free}")

b = 1
