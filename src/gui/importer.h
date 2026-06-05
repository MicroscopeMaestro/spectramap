#ifndef IMPORTER_H
#define IMPORTER_H

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace spmap {

struct Dataset {
    std::string name;
    std::vector<double> wavenumbers;
    Eigen::MatrixXd data; // N x M
    Eigen::MatrixXd position; // N x 2 or N x 3
    std::vector<std::string> labels;
    int m_grid = 1;
    int n_grid = 1;
    int l_grid = 1;
};

bool load_dataset(const std::string& filepath, const std::string& data_type, Dataset& out_dataset);

} // namespace spmap

#endif // IMPORTER_H
