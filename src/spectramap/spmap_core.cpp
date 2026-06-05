#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <thread>
#include <iostream>

namespace py = pybind11;

// Helper to find the median of a vector in O(N) time
double get_median(std::vector<double> v) {
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

// Solves A * x = d, where A is a tridiagonal matrix with:
// lower diag: a (size m-1, value -lambda)
// main diag: b (size m, value w_i + diag_lambda_i)
// upper diag: c (size m-1, value -lambda)
void tridiag_solve(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c, const std::vector<double>& d, std::vector<double>& x) {
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

// Single pixel airPLS baseline correction
void airpls_single(const double* x, int m, double landa, int itermax, double* z) {
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

    // Precompute diagonals of lambda * D.T * D
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

// Single spectrum SNIP baseline correction
void snip_single(const double* x, int m, int niter, double* z) {
    std::vector<double> temp(m);
    for (int i = 0; i < m; ++i) {
        temp[i] = std::log(std::log(std::sqrt(x[i] + 1.0) + 1.0) + 1.0);
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

// Single spectrum 1D Gaussian smoothing
void gaussian_single(const double* x, int m, double sigma, double* z) {
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
void fixer_single(const double* x, int m, int half_win, double limit, double* z) {
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

// Multithreaded execution wrapper
template<typename Func>
void run_parallel(int n_pixels, Func&& func) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::vector<std::thread> threads;
    int chunk_size = n_pixels / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? n_pixels : (t + 1) * chunk_size;
        if (start < end) {
            threads.emplace_back([=, &func]() {
                for (int p = start; p < end; ++p) {
                    func(p);
                }
            });
        }
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

// Parallel airPLS baseline correction (input: N x M, output: N x M)
py::array_t<double> airpls_parallel(py::array_t<double> matrix, double landa, int itermax) {
    py::buffer_info info = matrix.request();
    int n_pixels = info.shape[0];
    int m = info.shape[1];
    double* ptr = static_cast<double*>(info.ptr);

    auto result = py::array_t<double>({n_pixels, m});
    py::buffer_info res_info = result.request();
    double* res_ptr = static_cast<double*>(res_info.ptr);

    run_parallel(n_pixels, [&](int p) {
        airpls_single(ptr + p * m, m, landa, itermax, res_ptr + p * m);
    });

    return result;
}

// Parallel SNIP baseline correction (input: N x M, output: N x M)
py::array_t<double> snip_parallel(py::array_t<double> matrix, int niter) {
    py::buffer_info info = matrix.request();
    int n_pixels = info.shape[0];
    int m = info.shape[1];
    double* ptr = static_cast<double*>(info.ptr);

    auto result = py::array_t<double>({n_pixels, m});
    py::buffer_info res_info = result.request();
    double* res_ptr = static_cast<double*>(res_info.ptr);

    run_parallel(n_pixels, [&](int p) {
        snip_single(ptr + p * m, m, niter, res_ptr + p * m);
    });

    return result;
}

// Parallel 1D Gaussian smoothing (input: N x M, output: N x M)
py::array_t<double> gaussian_parallel(py::array_t<double> matrix, double sigma) {
    py::buffer_info info = matrix.request();
    int n_pixels = info.shape[0];
    int m = info.shape[1];
    double* ptr = static_cast<double*>(info.ptr);

    auto result = py::array_t<double>({n_pixels, m});
    py::buffer_info res_info = result.request();
    double* res_ptr = static_cast<double*>(res_info.ptr);

    run_parallel(n_pixels, [&](int p) {
        gaussian_single(ptr + p * m, m, sigma, res_ptr + p * m);
    });

    return result;
}

// Parallel Vector Normalization (L2 norm) (input: N x M, output: N x M)
py::array_t<double> vector_parallel(py::array_t<double> matrix) {
    py::buffer_info info = matrix.request();
    int n_pixels = info.shape[0];
    int m = info.shape[1];
    double* ptr = static_cast<double*>(info.ptr);

    auto result = py::array_t<double>({n_pixels, m});
    py::buffer_info res_info = result.request();
    double* res_ptr = static_cast<double*>(res_info.ptr);

    run_parallel(n_pixels, [&](int p) {
        const double* src = ptr + p * m;
        double* dst = res_ptr + p * m;
        double sum_sq = 0.0;
        for (int i = 0; i < m; ++i) {
            sum_sq += src[i] * src[i];
        }
        double norm = std::sqrt(sum_sq);
        if (norm > 1e-12) {
            for (int i = 0; i < m; ++i) {
                dst[i] = src[i] / norm;
            }
        } else {
            std::fill(dst, dst + m, 0.0);
        }
    });

    return result;
}

// Parallel Cosmic Ray Fixer (input: N x M, output: N x M)
py::array_t<double> fixer_parallel(py::array_t<double> matrix, int half_win, double limit) {
    py::buffer_info info = matrix.request();
    int n_pixels = info.shape[0];
    int m = info.shape[1];
    double* ptr = static_cast<double*>(info.ptr);

    auto result = py::array_t<double>({n_pixels, m});
    py::buffer_info res_info = result.request();
    double* res_ptr = static_cast<double*>(res_info.ptr);

    run_parallel(n_pixels, [&](int p) {
        fixer_single(ptr + p * m, m, half_win, limit, res_ptr + p * m);
    });

    return result;
}

// Parallel Normalization by Max value of each spectrum (input: N x M, output: N x M)
py::array_t<double> norm_parallel(py::array_t<double> matrix) {
    py::buffer_info info = matrix.request();
    int n_pixels = info.shape[0];
    int m = info.shape[1];
    double* ptr = static_cast<double*>(info.ptr);

    auto result = py::array_t<double>({n_pixels, m});
    py::buffer_info res_info = result.request();
    double* res_ptr = static_cast<double*>(res_info.ptr);

    run_parallel(n_pixels, [&](int p) {
        const double* src = ptr + p * m;
        double* dst = res_ptr + p * m;
        double max_val = src[0];
        for (int i = 1; i < m; ++i) {
            if (src[i] > max_val) max_val = src[i];
        }
        if (std::abs(max_val) > 1e-12) {
            for (int i = 0; i < m; ++i) {
                dst[i] = src[i] / max_val;
            }
        } else {
            std::fill(dst, dst + m, 0.0);
        }
    });

    return result;
}

// Parallel Normalization at Peak (input: N x M, output: N x M)
py::array_t<double> norm_at_peak_parallel(py::array_t<double> matrix, int peak_index) {
    py::buffer_info info = matrix.request();
    int n_pixels = info.shape[0];
    int m = info.shape[1];
    double* ptr = static_cast<double*>(info.ptr);

    auto result = py::array_t<double>({n_pixels, m});
    py::buffer_info res_info = result.request();
    double* res_ptr = static_cast<double*>(res_info.ptr);

    if (peak_index < 0 || peak_index >= m) {
        throw std::out_of_range("Peak index is out of bounds");
    }

    run_parallel(n_pixels, [&](int p) {
        const double* src = ptr + p * m;
        double* dst = res_ptr + p * m;
        double val = src[peak_index];
        if (std::abs(val) > 1e-12) {
            for (int i = 0; i < m; ++i) {
                dst[i] = src[i] / val;
            }
        } else {
            std::fill(dst, dst + m, 0.0);
        }
    });

    return result;
}

// Parallel NNLS abundance mapping (input: N x M, endmembers: C x M, output: N x C)
py::array_t<double> nnls_parallel(py::array_t<double> matrix, py::array_t<double> endmembers, int max_iter) {
    py::buffer_info mat_info = matrix.request();
    py::buffer_info em_info = endmembers.request();

    int N = mat_info.shape[0];
    int L = mat_info.shape[1];
    int C = em_info.shape[0];

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> M(static_cast<double*>(mat_info.ptr), N, L);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> U(static_cast<double*>(em_info.ptr), C, L);

    auto result = py::array_t<double>({N, C});
    py::buffer_info res_info = result.request();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> S(static_cast<double*>(res_info.ptr), N, C);

    // Precompute matrices
    Eigen::MatrixXd MUT = M * U.transpose(); // N x C
    Eigen::MatrixXd UUT = U * U.transpose(); // C x C

    S.fill(1.0 / C);

    // Run multiplicative updates
    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::MatrixXd SUUT = S * UUT; // N x C
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < C; ++j) {
                double denom = SUUT(i, j);
                if (denom > 1e-12 && MUT(i, j) > 0.0) {
                    S(i, j) *= MUT(i, j) / denom;
                } else if (MUT(i, j) <= 0.0) {
                    S(i, j) = 0.0;
                }
            }
        }
    }

    return result;
}

// Parallel OLS abundance mapping (input: N x M, endmembers: C x M, output: N x C)
py::array_t<double> ols_parallel(py::array_t<double> matrix, py::array_t<double> endmembers) {
    py::buffer_info mat_info = matrix.request();
    py::buffer_info em_info = endmembers.request();

    int N = mat_info.shape[0];
    int L = mat_info.shape[1];
    int C = em_info.shape[0];

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> M(static_cast<double*>(mat_info.ptr), N, L);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> U(static_cast<double*>(em_info.ptr), C, L);

    auto result = py::array_t<double>({N, C});
    py::buffer_info res_info = result.request();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> S(static_cast<double*>(res_info.ptr), N, C);

    Eigen::MatrixXd UT = U.transpose(); // L x C
    auto qr = UT.colPivHouseholderQr();

    run_parallel(N, [&](int i) {
        Eigen::VectorXd x = M.row(i).transpose();
        S.row(i) = qr.solve(x).transpose();
    });

    return result;
}

PYBIND11_MODULE(spmap_core, m) {
    m.def("airpls_parallel", &airpls_parallel, "Parallel airPLS baseline correction",
          py::arg("matrix"), py::arg("landa"), py::arg("itermax") = 50);
    m.def("snip_parallel", &snip_parallel, "Parallel SNIP baseline correction",
          py::arg("matrix"), py::arg("niter"));
    m.def("gaussian_parallel", &gaussian_parallel, "Parallel 1D Gaussian smoothing",
          py::arg("matrix"), py::arg("sigma"));
    m.def("vector_parallel", &vector_parallel, "Parallel L2 Vector Normalization",
          py::arg("matrix"));
    m.def("fixer_parallel", &fixer_parallel, "Parallel cosmic ray spike fixer",
          py::arg("matrix"), py::arg("half_win"), py::arg("limit"));
    m.def("norm_parallel", &norm_parallel, "Parallel normalization by max intensity",
          py::arg("matrix"));
    m.def("norm_at_peak_parallel", &norm_at_peak_parallel, "Parallel normalization at specific peak index",
          py::arg("matrix"), py::arg("peak_index"));
    m.def("nnls_parallel", &nnls_parallel, "Parallel NNLS abundance unmixing (Multiplicative Updates)",
          py::arg("matrix"), py::arg("endmembers"), py::arg("max_iter") = 500);
    m.def("ols_parallel", &ols_parallel, "Parallel OLS abundance unmixing",
          py::arg("matrix"), py::arg("endmembers"));
}
