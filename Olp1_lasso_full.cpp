#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <sstream>
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
    std::vector<std::vector<double>> data;
    std::ifstream file("OnlineNewsPopularity.csv");
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string val_str;
        std::getline(ss, val_str, ',');
        while (std::getline(ss, val_str, ',')) {
            try {
                double val = std::stod(val_str);
                row.push_back(val);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << val_str << std::endl;
            }
        }
        data.push_back(row);
    }
    int n = data.size();
    int p = data[0].size() - 1;
    MatrixXd X(n, p);
    VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) X(i, j) = data[i][j];
        y(i) = data[i][p];
    }
    X = X.rowwise() - X.colwise().mean();
    VectorXd std_dev = ((X.array().square().colwise().sum() / (n - 1)).sqrt());
    X = X.array().rowwise() / std_dev.transpose().array();
    y = y.array() - y.mean();
    auto start = std::chrono::high_resolution_clock::now();
    VectorXd beta_est = lasso_coordinate_descent(X, y);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Estimated coefficients:\n" << beta_est.transpose() << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    return 0;
}