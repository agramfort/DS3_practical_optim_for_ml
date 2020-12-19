
from numpy.linalg import norm

def sigmoid(t):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-t))

def cd_logreg(X, y, lamb, n_iter):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    Xw = X.dot(w)
    b = 0
    all_objs = np.empty(n_iter)
    lips_const = (np.linalg.norm(X, ord=2, axis=0) ** 2) / 4.
    
    for t in range(n_iter):
        for j in range(n_features + 1):
            if j < n_features:
                old_w_j = w[j]
                grad_j = np.sum(- y * X[:, j] / (1 + np.exp(y * (Xw + b)))) + lamb * w[j]
                w[j] -= grad_j / lips_const[j]
                Xw += X[:, j] * (w[j] - old_w_j)
            if j == n_features:
                old_b = b
                grad_b = np.sum(- y / (1 + np.exp(y * (Xw + b))))
                b -= grad_b * 4 / n_samples
            
        all_objs[t] = np.log(1. + np.exp(-y * (Xw + b))).sum() + lamb * norm(w, ord=2) ** 2
    
    return w, b, all_objs

w_hat, b_hat, all_objs = cd_logreg(X, y, lamb, n_iter=200)

plt.plot(all_objs)
plt.show()

plt.scatter(X[y > 0, 0], X[y > 0, 1], color='r')
plt.scatter(X[y < 0, 0], X[y < 0, 1], color='b')
xx = np.linspace(4, 8, 10)
plt.plot(xx,  - xx * w_hat[0] / w_hat[1] - b_hat / w_hat[1], 'k');
