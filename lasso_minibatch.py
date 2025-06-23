import time
import numpy as np


def soft_thresholding(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def lasso_coordinate_descent(X, y, lambda_=0.1, num_epoch=200, tol=1e-4, lr=1.0):
    n_samples, n_features = X.shape
    beta = np.random.random(n_features)
    batch_size = 20

    for epoch in range(num_epoch):
        #beta_old = beta.copy()
        row_number = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            beta_old = beta.copy() # 
            batch_pick = row_number[i : i + batch_size]
            X_random = X[batch_pick]
            residual_random = y[batch_pick] - X[batch_pick] @ beta
            max_delta = 0

            for j in range(n_features):
                gk = -np.dot(X_random[:, j], residual_random) / n_samples 
                beta[j] = beta_old[j] - lr * gk
                beta[j] = soft_thresholding(beta[j], lambda_)
                delta = beta[j] - beta_old[j]
                if np.abs(delta) > max_delta:          
                    max_delta = np.abs(delta)         
                residual_random -= X_random[:, j] * delta

            if max_delta < tol:          # change this convergence checking to the same as glmnet
                print(f"Converged in {epoch + 1} epochs.")
                return beta
        if epoch >= num_epoch - 3:
            print(f"Epoch {epoch + 1} beta[:10]:", np.round(beta[:10], 4))

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