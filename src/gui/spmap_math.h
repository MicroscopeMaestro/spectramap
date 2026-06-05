#ifndef SPMAP_MATH_H
#define SPMAP_MATH_H

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace spmap {

// Preprocessing
void airpls(const Eigen::MatrixXd& X, double lambda, int itermax, Eigen::MatrixXd& out_baselines);
void snip(const Eigen::MatrixXd& X, int niter, Eigen::MatrixXd& out_baselines);
void gaussian_smooth(const Eigen::MatrixXd& X, double sigma, Eigen::MatrixXd& out_smoothed);
void vector_normalize(const Eigen::MatrixXd& X, Eigen::MatrixXd& out_normed);
void fix_spikes(const Eigen::MatrixXd& X, int half_win, double limit, Eigen::MatrixXd& out_fixed);
void norm_max(const Eigen::MatrixXd& X, Eigen::MatrixXd& out_normed);
void norm_peak(const Eigen::MatrixXd& X, int peak_index, Eigen::MatrixXd& out_normed);

// PCA
void run_pca(const Eigen::MatrixXd& X, int n_components, Eigen::MatrixXd& out_scores, Eigen::MatrixXd& out_loadings);

// K-Means Clustering
void run_kmeans(const Eigen::MatrixXd& X, int k, std::vector<int>& out_labels, Eigen::MatrixXd& out_centroids);

// Abundance Unmixing
void run_nnls(const Eigen::MatrixXd& X, const Eigen::MatrixXd& endmembers, int max_iter, Eigen::MatrixXd& out_abundances);
void run_ols(const Eigen::MatrixXd& X, const Eigen::MatrixXd& endmembers, Eigen::MatrixXd& out_abundances);

} // namespace spmap

#endif // SPMAP_MATH_H
