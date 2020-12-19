
import matplotlib.pyplot as plt
import torch
from torch import nn

dtype = torch.float
device = torch.device("cpu")

lambd = 0.1

def f(w):
    pobj = np.sum(np.log(1. + np.exp(- y * np.dot(X, w)))) + lamdb / 2. * np.sum(w**2)
    return pobj

def f_grad_torch(w, X=X, y=y):
    X = torch.Tensor(X).to(device, dtype=dtype)
    y = torch.Tensor(y).to(device, dtype=dtype)
    w = torch.Tensor(w).to(device, dtype=dtype)
    w.requires_grad = True
    loss = torch.log(1. + torch.exp(-y * (X @ w))).sum() + lambd / 2. * (w ** 2).sum()
    loss.backward()
    return w.grad.detach().numpy().ravel()

w0 = np.random.randn(X.shape[1])
check_grad(f, f_grad_torch, w0)
