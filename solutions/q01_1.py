
from scipy.optimize import check_grad
import numpy as np

lamdb = 0.1

def f(w):
    pobj = np.sum(np.log(1. + np.exp(- y * np.dot(X, w)))) + lamdb / 2. * np.sum(w**2)
    return pobj

def f_grad(w):
    ywTx = y * np.dot(X, w)
    temp = 1. / (1. + np.exp(ywTx))
    grad = -np.dot(X.T, (y * temp)) + lamdb * w
    return grad

w0 = np.random.randn(X.shape[1])
check_grad(f, f_grad, w0)
