#include "spmap_math.h"
#include <cmath>
#include <algorithm>
#include <thread>
#include <numeric>
#include <random>
#include <iostream>

namespace spmap {

// Helper to run tasks in parallel
template<typename Func>
static void parallel_for(int n, Func&& func) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::vector<std::thread> threads;
    int chunk_size = n / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? n : (t + 1) * chunk_size;
        if (start < end) {
            threads.emplace_back([=, &func]() {
                for (int i = start; i < end; ++i) {
                    func(i);
                }
            });
        }
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

// Thomas algorithm solver for tridiagonal systems
static void tridiag_solve(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c, const std::vector<double>& d, std::vector<double>& x) {
    int n = d.size();
    if (n == 0) return;
    std::vector<double> c_prime(n, 0.0);
    std::vector<double> d_prime(n, 0.0);

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < n - 1; ++i) {
        double m = 1.0 / (b[i] - a[i - 1] * c_prime[i - 1]);
        c_prime[i] = c[i] * m;
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) * m;
    }

    double m = 1.0 / (b[n - 1] - a[n - 2] * c_prime[n - 2]);
    d_prime[n - 1] = (d[n - 1] - a[n - 2] * d_prime[n - 2]) * m;

    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
}

static double get_median(std::vector<double> v) {
    size_t n = v.size();
    if (n == 0) return 0.0;
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    double med = v[n / 2];
    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
        med = (med + v[n / 2 - 1]) / 2.0;
    }
    return med;
}

// Single spectrum airPLS
static void airpls_single(const double* x, int m, double landa, int itermax, double* z) {
    double min_x = x[0], max_x = x[0];
    double abs_x_sum = 0.0;
    for (int i = 0; i < m; ++i) {
        if (x[i] < min_x) min_x = x[i];
        if (x[i] > max_x) max_x = x[i];
        abs_x_sum += std::abs(x[i]);
    }
    if (max_x - min_x < 1e-12) {
        std::fill(z, z + m, 0.0);
        return;
    }

    std::vector<double> diag_lambda(m);
    std::vector<double> lower_lambda(m - 1, -landa);
    for (int i = 0; i < m; ++i) {
        if (i == 0 || i == m - 1) {
            diag_lambda[i] = landa;
        } else {
            diag_lambda[i] = 2.0 * landa;
        }
    }

    std::vector<double> w(m, 1.0);
    std::vector<double> b(m);
    std::vector<double> rhs(m);
    std::vector<double> z_vec(m);
    std::vector<double> d(m);

    for (int iter = 1; iter <= itermax; ++iter) {
        for (int i = 0; i < m; ++i) {
            b[i] = w[i] + diag_lambda[i];
            rhs[i] = w[i] * x[i];
        }

        tridiag_solve(lower_lambda, b, lower_lambda, rhs, z_vec);

        double d_neg_sum = 0.0;
        double d_neg_max = 0.0;
        int count_neg = 0;

        for (int i = 0; i < m; ++i) {
            d[i] = x[i] - z_vec[i];
            if (d[i] < 0) {
                double abs_d = std::abs(d[i]);
                d_neg_sum += abs_d;
                if (abs_d > d_neg_max) {
                    d_neg_max = abs_d;
                }
                count_neg++;
            }
        }

        double dssn = (count_neg > 0) ? d_neg_sum : 1.0;

        if (dssn < 0.001 * abs_x_sum || iter == itermax) {
            std::copy(z_vec.begin(), z_vec.end(), z);
            break;
        }

        for (int i = 0; i < m; ++i) {
            if (d[i] >= 0) {
                w[i] = 0.0;
            } else {
                double exponent = iter * std::abs(d[i]) / dssn;
                if (exponent > 500.0) exponent = 500.0;
                w[i] = std::exp(exponent);
            }
        }

        if (count_neg > 0) {
            double exponent = iter * d_neg_max / dssn;
            if (exponent > 500.0) exponent = 500.0;
            w[0] = std::exp(exponent);
            w[m - 1] = w[0];
        }
    }
}

// Single spectrum SNIP
static void snip_single(const double* x, int m, int niter, double* z) {
    std::vector<double> temp(m);
    for (int i = 0; i < m; ++i) {
        temp[i] = std::log(std::log(std::sqrt(std::max(0.0, x[i]) + 1.0) + 1.0) + 1.0);
    }
    for (int pp = 1; pp <= niter; ++pp) {
        std::vector<double> prev = temp;
        for (int i = pp; i < m - pp; ++i) {
            double r2 = (prev[i - pp] + prev[i + pp]) / 2.0;
            temp[i] = std::min(prev[i], r2);
        }
    }
    for (int i = 0; i < m; ++i) {
        double exp1 = std::exp(temp[i]) - 1.0;
        double exp2 = std::exp(exp1) - 1.0;
        z[i] = exp2 * exp2 - 1.0;
    }
}

// Single spectrum Gaussian filter
static void gaussian_single(const double* x, int m, double sigma, double* z) {
    if (sigma <= 0.0) {
        std::copy(x, x + m, z);
        return;
    }
    int radius = static_cast<int>(std::ceil(4.0 * sigma));
    int kernel_size = 2 * radius + 1;
    std::vector<double> kernel(kernel_size);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        kernel[i + radius] = std::exp(-(i * i) / (2.0 * sigma * sigma));
        sum += kernel[i + radius];
    }
    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] /= sum;
    }

    for (int i = 0; i < m; ++i) {
        double val = 0.0;
        for (int k = -radius; k <= radius; ++k) {
            int idx = i + k;
            if (idx < 0) {
                idx = -idx - 1;
            } else if (idx >= m) {
                idx = 2 * m - idx - 1;
            }
            if (idx < 0) idx = 0;
            if (idx >= m) idx = m - 1;
            val += x[idx] * kernel[k + radius];
        }
        z[i] = val;
    }
}

// Single spectrum cosmic ray fixer
static void fixer_single(const double* x, int m, int half_win, double limit, double* z) {
    std::copy(x, x + m, z);
    if (m <= 2) return;

    std::vector<double> diff_x(m - 1);
    for (int i = 0; i < m - 1; ++i) {
        diff_x[i] = x[i + 1] - x[i];
    }

    double median_diff = get_median(diff_x);
    std::vector<double> abs_diff(m - 1);
    for (int i = 0; i < m - 1; ++i) {
        abs_diff[i] = std::abs(diff_x[i] - median_diff);
    }
    double mad_diff = get_median(abs_diff);
    if (mad_diff < 1e-12) mad_diff = 1e-12;

    std::vector<bool> spikes(m - 1, false);
    for (int i = 0; i < m - 1; ++i) {
        double modified_z = 0.6745 * (diff_x[i] - median_diff) / mad_diff;
        if (std::abs(modified_z) > limit) {
            spikes[i] = true;
        }
    }

    for (int i = half_win; i < m - 1 - half_win; ++i) {
        if (spikes[i]) {
            std::vector<int> valid_indices;
            for (int k = -half_win; k <= half_win; ++k) {
                int w_idx = i + k;
                if (w_idx >= 0 && w_idx < m - 1 && !spikes[w_idx]) {
                    valid_indices.push_back(w_idx);
                }
            }
            if (!valid_indices.empty()) {
                double sum = 0.0;
                for (int idx : valid_indices) {
                    sum += x[idx];
                }
                z[i] = sum / valid_indices.size();
            }
        }
    }
}

// Preprocessing implementations
void airpls(const Eigen::MatrixXd& X, double lambda, int itermax, Eigen::MatrixXd& out_baselines) {
    int n = X.rows();
    int m = X.cols();
    out_baselines.resize(n, m);
    parallel_for(n, [&](int i) {
        airpls_single(X.row(i).data(), m, lambda, itermax, out_baselines.row(i).data());
    });
}

void snip(const Eigen::MatrixXd& X, int niter, Eigen::MatrixXd& out_baselines) {
    int n = X.rows();
    int m = X.cols();
    out_baselines.resize(n, m);
    parallel_for(n, [&](int i) {
        snip_single(X.row(i).data(), m, niter, out_baselines.row(i).data());
    });
}

void gaussian_smooth(const Eigen::MatrixXd& X, double sigma, Eigen::MatrixXd& out_smoothed) {
    int n = X.rows();
    int m = X.cols();
    out_smoothed.resize(n, m);
    parallel_for(n, [&](int i) {
        gaussian_single(X.row(i).data(), m, sigma, out_smoothed.row(i).data());
    });
}

void vector_normalize(const Eigen::MatrixXd& X, Eigen::MatrixXd& out_normed) {
    int n = X.rows();
    int m = X.cols();
    out_normed.resize(n, m);
    parallel_for(n, [&](int i) {
        double norm = X.row(i).norm();
        if (norm > 1e-12) {
            out_normed.row(i) = X.row(i) / norm;
        } else {
            out_normed.row(i).setZero();
        }
    });
}

void fix_spikes(const Eigen::MatrixXd& X, int half_win, double limit, Eigen::MatrixXd& out_fixed) {
    int n = X.rows();
    int m = X.cols();
    out_fixed.resize(n, m);
    parallel_for(n, [&](int i) {
        fixer_single(X.row(i).data(), m, half_win, limit, out_fixed.row(i).data());
    });
}

void norm_max(const Eigen::MatrixXd& X, Eigen::MatrixXd& out_normed) {
    int n = X.rows();
    int m = X.cols();
    out_normed.resize(n, m);
    parallel_for(n, [&](int i) {
        double max_val = X.row(i).maxCoeff();
        if (std::abs(max_val) > 1e-12) {
            out_normed.row(i) = X.row(i) / max_val;
        } else {
            out_normed.row(i).setZero();
        }
    });
}

void norm_peak(const Eigen::MatrixXd& X, int peak_index, Eigen::MatrixXd& out_normed) {
    int n = X.rows();
    int m = X.cols();
    out_normed.resize(n, m);
    parallel_for(n, [&](int i) {
        double val = X(i, peak_index);
        if (std::abs(val) > 1e-12) {
            out_normed.row(i) = X.row(i) / val;
        } else {
            out_normed.row(i).setZero();
        }
    });
}

// PCA implementation
void run_pca(const Eigen::MatrixXd& X, int n_components, Eigen::MatrixXd& out_scores, Eigen::MatrixXd& out_loadings) {
    int n = X.rows();
    int m = X.cols();
    if (n_components > m) n_components = m;
    
    // Centering data columns
    Eigen::VectorXd mean = X.colwise().mean();
    Eigen::MatrixXd X_centered = X.rowwise() - mean.transpose();
    
    // Covariance matrix
    Eigen::MatrixXd Cov = (X_centered.transpose() * X_centered) / (n - 1);
    
    // Eigen Decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Cov);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();
    
    // Sort eigenvectors by eigenvalues descending
    out_loadings.resize(m, n_components);
    for (int i = 0; i < n_components; ++i) {
        out_loadings.col(i) = eigenvectors.col(m - 1 - i);
    }
    
    // Project centered data to get scores
    out_scores = X_centered * out_loadings;
}

// K-Means implementation
void run_kmeans(const Eigen::MatrixXd& X, int k, std::vector<int>& out_labels, Eigen::MatrixXd& out_centroids) {
    int n = X.rows();
    int m = X.cols();
    out_labels.assign(n, 0);
    out_centroids.resize(k, m);
    
    if (n == 0 || k <= 0) return;
    
    // Random initialization of centroids
    std::mt19937 rng(42); // fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < k; ++i) {
        out_centroids.row(i) = X.row(dist(rng));
    }
    
    std::vector<int> cluster_sizes(k, 0);
    bool changed = true;
    int max_iters = 100;
    
    for (int iter = 0; iter < max_iters && changed; ++iter) {
        changed = false;
        
        // Step 1: Assignment
        parallel_for(n, [&](int i) {
            double min_dist = 1e30;
            int best_cluster = 0;
            for (int c = 0; c < k; ++c) {
                double d = (X.row(i) - out_centroids.row(c)).squaredNorm();
                if (d < min_dist) {
                    min_dist = d;
                    best_cluster = c;
                }
            }
            if (out_labels[i] != best_cluster) {
                out_labels[i] = best_cluster;
                changed = true;
            }
        });
        
        // Step 2: Centroid Update
        out_centroids.setZero();
        cluster_sizes.assign(k, 0);
        for (int i = 0; i < n; ++i) {
            int c = out_labels[i];
            out_centroids.row(c) += X.row(i);
            cluster_sizes[c]++;
        }
        for (int c = 0; c < k; ++c) {
            if (cluster_sizes[c] > 0) {
                out_centroids.row(c) /= cluster_sizes[c];
            } else {
                // If a cluster becomes empty, reinitialize to a random point
                out_centroids.row(c) = X.row(dist(rng));
            }
        }
    }
}

// NNLS implementation (Multiplicative Updates)
void run_nnls(const Eigen::MatrixXd& X, const Eigen::MatrixXd& endmembers, int max_iter, Eigen::MatrixXd& out_abundances) {
    int N = X.rows();
    int L = X.cols();
    int C = endmembers.rows();

    out_abundances.resize(N, C);
    out_abundances.fill(1.0 / C);

    Eigen::MatrixXd MUT = X * endmembers.transpose(); // N x C
    Eigen::MatrixXd UUT = endmembers * endmembers.transpose(); // C x C

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::MatrixXd SUUT = out_abundances * UUT; // N x C
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < C; ++j) {
                double denom = SUUT(i, j);
                if (denom > 1e-12 && MUT(i, j) > 0.0) {
                    out_abundances(i, j) *= MUT(i, j) / denom;
                } else if (MUT(i, j) <= 0.0) {
                    out_abundances(i, j) = 0.0;
                }
            }
        }
    }
}

// OLS implementation
void run_ols(const Eigen::MatrixXd& X, const Eigen::MatrixXd& endmembers, Eigen::MatrixXd& out_abundances) {
    int N = X.rows();
    int L = X.cols();
    int C = endmembers.rows();

    out_abundances.resize(N, C);
    Eigen::MatrixXd UT = endmembers.transpose(); // L x C
    auto qr = UT.colPivHouseholderQr();

    parallel_for(N, [&](int i) {
        Eigen::VectorXd x = X.row(i).transpose();
        out_abundances.row(i) = qr.solve(x).transpose();
    });
}

} // namespace spmap
