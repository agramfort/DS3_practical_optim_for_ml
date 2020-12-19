
Xb = np.concatenate((X, np.ones((len(X), 1))), axis=1)

def f(w):
    pobj = np.sum(np.log(1. + np.exp(- y * (np.dot(X, w[:-1]) + w[-1]))))
    pobj += lambd * np.dot(w[:-1], w[:-1]) / 2.
    return pobj

def f_grad(w):
    ywTx = y * np.dot(Xb, w)
    temp = 1. / (1. + np.exp(ywTx))
    grad = -np.dot(Xb.T, (y * temp)) + lambd * w
    grad[-1] -= lambd * w[-1]  # take care of intercept
    return grad

from scipy.optimize import fmin_l_bfgs_b
w, _, _ = fmin_l_bfgs_b(f, x0=np.zeros(X.shape[1] + 1), fprime=f_grad)


plt.scatter(X[y > 0, 0], X[y > 0, 1], color='r')
plt.scatter(X[y < 0, 0], X[y < 0, 1], color='b')
xx = np.linspace(4, 8, 10)
plt.plot(xx,  - xx * w[0] / w[1] - w[2] / w[1], 'k');
