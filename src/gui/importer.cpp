#include "importer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <memory>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

namespace spmap {

bool load_dataset(const std::string& filepath, const std::string& data_type, Dataset& out_dataset) {
    // Generate paths
    std::string temp_csv = "data/temp_extracted.csv";
    std::string temp_meta = temp_csv + ".meta";
    
    // Ensure data directory exists
    std::string data_dir = "data";
    // Command to run python importer
    std::string cmd = "python tools/gui_importer.py \"" + filepath + "\" \"" + data_type + "\" \"" + temp_csv + "\"";
    std::cout << "Running importer command: " << cmd << std::endl;
    
    // Run the command
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        std::cerr << "Failed to run python importer command." << std::endl;
        return false;
    }
    
    char buffer[256];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        result += buffer;
    }
    
    std::cout << "Python importer output: " << result << std::endl;
    
    if (result.find("SUCCESS") == std::string::npos) {
        std::cerr << "Python importer failed to extract data." << std::endl;
        return false;
    }
    
    // Read meta file
    std::ifstream meta_f(temp_meta);
    if (!meta_f.is_open()) {
        std::cerr << "Failed to open metadata file: " << temp_meta << std::endl;
        return false;
    }
    
    std::string meta_line;
    std::getline(meta_f, meta_line);
    std::stringstream meta_ss(meta_line);
    std::string m_str, n_str, l_str, has_z_str;
    std::getline(meta_ss, m_str, ',');
    std::getline(meta_ss, n_str, ',');
    std::getline(meta_ss, l_str, ',');
    std::getline(meta_ss, has_z_str, ',');
    
    out_dataset.m_grid = std::stoi(m_str);
    out_dataset.n_grid = std::stoi(n_str);
    out_dataset.l_grid = std::stoi(l_str);
    bool has_z = std::stoi(has_z_str) != 0;
    
    // Parse CSV
    std::ifstream csv_f(temp_csv);
    if (!csv_f.is_open()) {
        std::cerr << "Failed to open temporary CSV: " << temp_csv << std::endl;
        return false;
    }
    
    std::string line;
    // Read headers
    if (!std::getline(csv_f, line)) {
        return false;
    }
    
    std::stringstream header_ss(line);
    std::string col_name;
    std::vector<std::string> headers;
    while (std::getline(header_ss, col_name, ',')) {
        headers.push_back(col_name);
    }
    
    int num_metadata_cols = has_z ? 4 : 3; // label, x, y, (z)
    int num_bands = headers.size() - num_metadata_cols;
    
    out_dataset.wavenumbers.clear();
    for (size_t i = num_metadata_cols; i < headers.size(); ++i) {
        out_dataset.wavenumbers.push_back(std::stod(headers[i]));
    }
    
    std::vector<std::string> labels;
    std::vector<std::vector<double>> positions_temp;
    std::vector<std::vector<double>> data_temp;
    
    while (std::getline(csv_f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        
        // label
        std::getline(ss, val, ',');
        labels.push_back(val);
        
        // x, y
        std::string x_str, y_str;
        std::getline(ss, x_str, ',');
        std::getline(ss, y_str, ',');
        double x_coord = std::stod(x_str);
        double y_coord = std::stod(y_str);
        
        if (has_z) {
            std::string z_str;
            std::getline(ss, z_str, ',');
            double z_coord = std::stod(z_str);
            positions_temp.push_back({x_coord, y_coord, z_coord});
        } else {
            positions_temp.push_back({x_coord, y_coord});
        }
        
        // intensities
        std::vector<double> row_data;
        while (std::getline(ss, val, ',')) {
            row_data.push_back(std::stod(val));
        }
        data_temp.push_back(row_data);
    }
    
    int num_pixels = data_temp.size();
    out_dataset.labels = labels;
    
    // Convert data to Eigen
    out_dataset.data.resize(num_pixels, num_bands);
    for (int i = 0; i < num_pixels; ++i) {
        // Just in case the row had slightly different number of columns
        int items = std::min(num_bands, (int)data_temp[i].size());
        for (int j = 0; j < items; ++j) {
            out_dataset.data(i, j) = data_temp[i][j];
        }
    }
    
    // Convert positions to Eigen
    int pos_dim = has_z ? 3 : 2;
    out_dataset.position.resize(num_pixels, pos_dim);
    for (int i = 0; i < num_pixels; ++i) {
        for (int j = 0; j < pos_dim; ++j) {
            out_dataset.position(i, j) = positions_temp[i][j];
        }
    }
    
    out_dataset.name = filepath;
    
    // Clean up temporary files
    meta_f.close();
    csv_f.close();
    std::remove(temp_csv.c_str());
    std::remove(temp_meta.c_str());
    
    return true;
}

} // namespace spmap
