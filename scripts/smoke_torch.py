import os
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.randn(2, 2, device="cuda")
    y = x @ x.t()
    print("cuda matmul ok:", y.shape)
else:
    print("running on cpu; install CUDA build if needed")
