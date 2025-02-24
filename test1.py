import torch
print(torch.cuda.is_available())  # 如果使用 NVIDIA GPU会返回 True
print(torch.xpu.is_available())   # 如果使用 Intel GPU会返回 True
print(torch.xpu.device_count())