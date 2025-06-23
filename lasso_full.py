import time
import numpy as np

import numpy as np

def soft_thresholding(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

def lasso_coordinate_descent(X, y, lambda_=0.1, max_iter=1000, tol=1e-4,lr=1.0):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    r = y - X @ beta  
    #X_norm_sq = np.sum(X ** 2, axis=0)/n_samples

    for iteration in range(max_iter):
        beta_old = beta.copy()

        for j in range(n_features):
            gk = -np.dot(X[:, j], r)/n_samples
            beta[j] = beta_old[j]-lr * gk
            beta[j] = soft_thresholding(beta[j], lambda_)
            delta = beta[j] - beta_old[j]
            r -= delta * X[:, j]

        if np.linalg.norm(beta - beta_old,ord=2) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break

    return beta





# Simulated data
np.random.seed(0)
n, p = 100, 1000
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = [2, -1, 0.5, 0, 3]  # sparse true coefficients
y = X @ true_beta + np.random.randn(n) * 0.5

# Standardize X and center y
X -= X.mean(axis=0)
X /= X.std(axis=0)
y -= y.mean()

# Fit LASSO
start = time.time()
beta_est = lasso_coordinate_descent(X, y)
end = time.time()

print("Estimated coefficients:\n", np.round(beta_est, 3))
print("Time: ", (end - start) * 1000, "ms")
