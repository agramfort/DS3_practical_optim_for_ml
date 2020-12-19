
def newton_logistic(X, y, lambd=1.):
    X = np.asarray(X, dtype=np.float)
    y = np.asarray(y, dtype=np.float)

    n_samples, n_features = X.shape
    X = np.concatenate((X, np.ones((len(X), 1))), axis=1)

    w = np.zeros(n_features + 1)
    pobj = []

    for k in range(20):
        ywTx = y * np.dot(X, w)
        temp = 1. / (1. + np.exp(ywTx))
        grad = - np.dot(X.T, (y * temp)) + lambd * w
        hess = np.dot(X.T, (temp * ( 1. - temp ))[:, None] * X)
        hess.flat[::n_features + 2] += lambd

        hess[-1, -1] -= lambd  # don't penalize intercept
        grad[-1] -= lambd * w[-1]

        w -= linalg.solve(hess, grad)

        this_pobj = np.sum(np.log(1. + np.exp( - y * np.dot(X, w))))
        this_pobj += lambd * np.dot(w[:-1], w[:-1]) / 2.
        pobj.append(this_pobj)

    print("Global minimum : %s" % pobj[-1])

    w, b = w[:-1], w[-1]
    return w, b, pobj


w, b, pobj = newton_logistic(X, y)

plt.plot(np.log10(pobj - pobj[-1] + np.finfo('float').eps), 'b')
plt.xlabel('Iterations')
plt.ylabel(r'$f(x^k) - f(x^*)$')
plt.show()

plt.scatter(X[y > 0, 0], X[y > 0, 1], color='r')
plt.scatter(X[y < 0, 0], X[y < 0, 1], color='b')
xx = np.linspace(4, 8, 10)
plt.plot(xx,  - xx * w[0] / w[1] - b / w[1], 'k');
