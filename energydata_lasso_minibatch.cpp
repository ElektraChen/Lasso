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

VectorXd lasso_coordinate_descent(const MatrixXd& X, const VectorXd& y, double lambda=0.1, int num_epoch=200, double tol=1e-4, double lr=1.0) {
    int n_samples = X.rows();
    int n_features = X.cols();
    //VectorXd beta = VectorXd::Random(n_features);
    VectorXd beta = VectorXd::Zero(n_features);
    int batch_size = 128;
    std::default_random_engine generator;
    
    for (int epoch = 0; epoch < num_epoch; ++epoch) {
        std::vector<int> indices(n_samples);
        for (int i = 0; i < n_samples; ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), generator);
        
        for (int i = 0; i < n_samples; i += batch_size) {
            VectorXd beta_old = beta;
            int current_batch_size = std::min(batch_size, n_samples - i);
            MatrixXd X_batch(current_batch_size, n_features);
            VectorXd y_batch(current_batch_size);
            for (int b = 0; b < current_batch_size; ++b) {
                int idx = indices[i + b];
                X_batch.row(b) = X.row(idx);
                y_batch(b) = y(idx);
            }
            
            VectorXd residual = y_batch - X_batch * beta;
            double max_delta = 0.0;
            
            for (int j = 0; j < n_features; ++j) {
                //double beta_old_j = beta(j);
                double gradient = -X_batch.col(j).dot(residual) / n_samples;
                beta(j) = beta_old(j) - lr * gradient;
                beta(j) = soft_thresholding(beta(j), lambda);
                double delta = beta(j) - beta_old(j);
                residual -= X_batch.col(j) * delta;
                if (std::abs(delta) > max_delta) {
                    max_delta = std::abs(delta);
                }
            }
            
            if (max_delta < tol) {
                std::cout << "Converged in " << epoch + 1 << " epochs." << std::endl;
                return beta;
            }
        }
    }
    std::cout << "Reached maximum epochs: " << num_epoch << std::endl;
    return beta;
}

int main() {
    std::vector<std::vector<double>> X_rows;
    std::vector<double>              y_vals;
    std::ifstream file("energydata_complete.csv");
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        std::getline(ss, token, ',');
        token.erase(std::remove(token.begin(), token.end(), '"'), token.end());
        double y_val = std::stod(token);

        std::vector<double> row;
        while (std::getline(ss, token, ',')) {
            token.erase(std::remove(token.begin(), token.end(), '"'), token.end());
            row.push_back(std::stod(token));
        }
        if (!row.empty()) {
        y_vals.push_back(y_val);
        X_rows.push_back(row);
        }
    }

    int n = X_rows.size();
    int p = X_rows[0].size();      
    MatrixXd X(n, p);
    VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) X(i, j) = X_rows[i][j];
        y(i) = y_vals[i];
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