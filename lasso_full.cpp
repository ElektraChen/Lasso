#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <chrono>

using namespace Eigen;

double soft_thresholding(double z, double gamma) {
    return std::copysign(1.0, z) * std::max(std::abs(z) - gamma, 0.0);
}

VectorXd lasso_coordinate_descent(const MatrixXd& X, const VectorXd& y, double lambda=0.1, int max_iter=1000, double tol=1e-4, double lr=1.0) {
    int n_samples = X.rows();
    int n_features = X.cols();
    //VectorXd beta = VectorXd::Random(n_features);
    VectorXd beta = VectorXd::Zero(n_features);
    VectorXd residual = y - X * beta;

    for (int iteration = 0; iteration < max_iter; ++iteration) {
        VectorXd beta_old = beta;
        double max_delta = 0.0;

        for (int j = 0; j < n_features; ++j) {
            double gradient = -X.col(j).dot(residual) / n_samples;
            beta(j) = beta_old(j) - lr * gradient;
            beta(j) = soft_thresholding(beta(j), lambda);
            double delta = beta(j) - beta_old(j);
            residual -= X.col(j) * delta;
            if (std::abs(delta) > max_delta) {
                max_delta = std::abs(delta);
            }
        }

        if (max_delta < tol) {
            std::cout << "Converged in " << iteration + 1 << " iterations." << std::endl;
            return beta;
        }
    }
    std::cout << "Reached maximum iterations: " << max_iter << std::endl;
    return beta;
}


int main() {
    std::srand(0);
    int n = 100, p = 1000;
    MatrixXd X(n, p);
    VectorXd y(n);
    VectorXd true_beta = VectorXd::Zero(p);
    true_beta(0) = 2.0;
    true_beta(1) = -1.0;
    true_beta(2) = 0.5;
    true_beta(4) = 3.0;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            X(i, j) = std::rand() / double(RAND_MAX);
        }
        y(i) = X.row(i).dot(true_beta) + (std::rand() / double(RAND_MAX)) * 0.5;
    }
    
    VectorXd X_mean = X.colwise().mean();
    X = X.rowwise() - X_mean.transpose();
    VectorXd X_std = ((X.array().square().colwise().sum() / (n - 1)).sqrt());
    X = X.array().rowwise() / X_std.transpose().array();
    y = y.array() - y.mean();
    
    auto start = std::chrono::high_resolution_clock::now();
    VectorXd beta_est = lasso_coordinate_descent(X, y);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Estimated coefficients:\n" << beta_est.transpose() << std::endl;
    std::cout << "\nTime: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    
    return 0;
}