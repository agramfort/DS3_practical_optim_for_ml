
def run_exp(n_samples, lambd, corr):
    # generate indices of random samples
    n_iter = 30
    iis = np.random.randint(0, n_samples, n_samples * n_iter)

    X, y = simulate(w_true, n_samples, std=1., corr=corr)

    w_init = np.zeros(n_features)
    w_min, f_min, _ = fmin_l_bfgs_b(f, w_init, f_grad,
                                    args=(X, y, lambd), pgtol=1e-30, factr=1e-30)

    step = 1. / lipschitz_logreg(X, y, lambd)
    w_init = np.zeros(n_features)
    monitor_gd = monitor(gd, f, w_min, (X, y, lambd))
    monitor_gd.run(w_init, f_grad, n_iter, step, args=(X, y, lambd))

    step0 = 1e-1
    w_init = np.zeros(n_features)

    monitor_sgd = monitor(sgd, f, w_min, (X, y, lambd))
    monitor_sgd.run(w_init, iis, f_grad_i, n_iter * n_samples, step0, n_samples,
                    args=(X, y, lambd))

    monitors = [monitor_gd, monitor_sgd]
    solvers = ["GD", "SGD"]
    plot_epochs(monitors, solvers)
    plt.suptitle(f'n_samples={n_samples} -- log10(lambda)={np.log10(lambd)} -- corr={corr}')


# Small n_samples vs large n_samples
run_exp(n_samples=1000, lambd=1 / 1000, corr=0.1)
run_exp(n_samples=100000, lambd=1 / 100000, corr=0.1)

# Small reg vs large reg
run_exp(n_samples=100000, lambd=1 / 100000, corr=0.1)
run_exp(n_samples=100000, lambd=1 / np.sqrt(100000), corr=0.1)

# Small corr vs large corr
run_exp(n_samples=100000, lambd=1 / 100000, corr=0.1)
run_exp(n_samples=100000, lambd=1 / 100000, corr=0.7)
