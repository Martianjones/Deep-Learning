import torch
print(torch.cuda.is_available())  # 如果输出 True，表示 CUDA 可用
print(torch.cuda.current_device())  # 输出当前设备的 ID
print(torch.cuda.get_device_name(0))  # 输出 GPU 名称
