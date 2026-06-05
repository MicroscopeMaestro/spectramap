#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cmath>

#include "importer.h"
#include "spmap_math.h"

namespace fs = std::filesystem;

// UI State
struct AppState {
    std::vector<std::string> data_files;
    int selected_file_idx = 0;
    std::string selected_data_type = "hyper_image";
    
    // Dataset states
    spmap::Dataset current_dataset;
    spmap::Dataset original_dataset; // for resetting
    bool is_loaded = false;
    bool is_loading = false;
    std::string load_error = "";
    
    // Preprocessing parameters
    float keep_min = 400.0f;
    float keep_max = 1850.0f;
    int snip_iter = 30;
    float gaussian_sigma = 2.0f;
    int spike_size = 3;
    float spike_limit = 7.0f;
    
    // Analysis parameters
    int analysis_type = 0; // 0: None, 1: PCA, 2: K-Means, 3: Linear Unmixing
    
    // PCA State
    int pca_components = 3;
    bool has_pca = false;
    Eigen::MatrixXd pca_scores;   // N x components
    Eigen::MatrixXd pca_loadings; // M x components
    
    // K-Means State
    int kmeans_k = 4;
    bool has_kmeans = false;
    std::vector<int> kmeans_labels;
    Eigen::MatrixXd kmeans_centroids;
    
    // Unmixing State
    int unmixing_constraint = 0; // 0: NNLS, 1: OLS
    bool has_unmixing = false;
    Eigen::MatrixXd abundances;
    
    // Visualization State
    int current_tab = 0;
    double selected_wavenumber = 1000.0;
    int selected_wavenumber_idx = 0;
    int selected_pixel_idx = -1;
    int selected_abundance_component = 0;
    int selected_pca_component = 0;
    
    // Heatmap data helper
    std::vector<double> heatmap_data;
    double heatmap_min = 0.0;
    double heatmap_max = 1.0;
};

// Style customization for a premium modern look
void apply_premium_style() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::StyleColorsDark();
    
    style.WindowRounding = 8.0f;
    style.FrameRounding = 5.0f;
    style.GrabRounding = 5.0f;
    style.PopupRounding = 5.0f;
    style.ScrollbarRounding = 5.0f;
    style.TabRounding = 5.0f;
    
    // Custom Slate/Purple/Teal color palette
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text]                   = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.09f, 0.09f, 0.12f, 0.94f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.12f, 0.12f, 0.16f, 0.00f);
    colors[ImGuiCol_Border]                 = ImVec4(0.20f, 0.20f, 0.27f, 1.00f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.15f, 0.16f, 0.21f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.22f, 0.23f, 0.31f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.28f, 0.29f, 0.40f, 1.00f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.12f, 0.12f, 0.17f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.16f, 0.17f, 0.24f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.56f, 0.45f, 0.90f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.56f, 0.45f, 0.90f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.66f, 0.55f, 0.95f, 1.00f);
    colors[ImGuiCol_Button]                 = ImVec4(0.23f, 0.20f, 0.38f, 1.00f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.35f, 0.30f, 0.60f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.45f, 0.40f, 0.75f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.24f, 0.23f, 0.32f, 1.00f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.30f, 0.29f, 0.42f, 1.00f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.38f, 0.37f, 0.55f, 1.00f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.15f, 0.16f, 0.22f, 1.00f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.35f, 0.30f, 0.60f, 1.00f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.23f, 0.20f, 0.38f, 1.00f);
}

// Scans for datasets
void refresh_data_files(AppState& state) {
    state.data_files.clear();
    if (!fs::exists("data")) {
        fs::create_directory("data");
    }
    for (const auto& entry : fs::directory_iterator("data")) {
        auto filename = entry.path().filename().string();
        if (filename.find(".csv.xz") != std::string::npos || filename.find(".spc") != std::string::npos) {
            state.data_files.push_back(filename);
        }
    }
    // Add default files if none found
    if (state.data_files.empty()) {
        state.data_files.push_back("3D.csv.xz");
        state.data_files.push_back("raman_sample.spc");
    }
}

// Helper to update the heatmap data to visualize spatial configuration
void update_heatmap(AppState& state) {
    if (!state.is_loaded) return;
    
    int n_pixels = state.current_dataset.data.rows();
    int m_grid = state.current_dataset.m_grid;
    int n_grid = state.current_dataset.n_grid;
    
    state.heatmap_data.assign(m_grid * n_grid, 0.0);
    
    // Choose what value to visualize based on analysis mode
    if (state.analysis_type == 2 && state.has_kmeans) {
        // Plot K-Means Labels
        for (int i = 0; i < n_pixels; ++i) {
            int x = static_cast<int>(state.current_dataset.position(i, 0));
            int y = static_cast<int>(state.current_dataset.position(i, 1));
            if (x >= 0 && x < m_grid && y >= 0 && y < n_grid) {
                state.heatmap_data[y * m_grid + x] = state.kmeans_labels[i];
            }
        }
        state.heatmap_min = 0.0;
        state.heatmap_max = state.kmeans_k - 1;
    } 
    else if (state.analysis_type == 3 && state.has_unmixing) {
        // Plot Abundances
        int comp = state.selected_abundance_component;
        for (int i = 0; i < n_pixels; ++i) {
            int x = static_cast<int>(state.current_dataset.position(i, 0));
            int y = static_cast<int>(state.current_dataset.position(i, 1));
            if (x >= 0 && x < m_grid && y >= 0 && y < n_grid) {
                state.heatmap_data[y * m_grid + x] = state.abundances(i, comp);
            }
        }
        state.heatmap_min = 0.0;
        state.heatmap_max = 1.0;
    } 
    else if (state.analysis_type == 1 && state.has_pca) {
        // Plot PCA score
        int comp = state.selected_pca_component;
        double min_val = 1e30, max_val = -1e30;
        for (int i = 0; i < n_pixels; ++i) {
            int x = static_cast<int>(state.current_dataset.position(i, 0));
            int y = static_cast<int>(state.current_dataset.position(i, 1));
            if (x >= 0 && x < m_grid && y >= 0 && y < n_grid) {
                double val = state.pca_scores(i, comp);
                state.heatmap_data[y * m_grid + x] = val;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
        state.heatmap_min = min_val;
        state.heatmap_max = max_val;
    } 
    else {
        // Plot single Wavenumber Intensity
        int w_idx = state.selected_wavenumber_idx;
        double min_val = 1e30, max_val = -1e30;
        for (int i = 0; i < n_pixels; ++i) {
            int x = static_cast<int>(state.current_dataset.position(i, 0));
            int y = static_cast<int>(state.current_dataset.position(i, 1));
            if (x >= 0 && x < m_grid && y >= 0 && y < n_grid) {
                double val = state.current_dataset.data(i, w_idx);
                state.heatmap_data[y * m_grid + x] = val;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
        state.heatmap_min = min_val;
        state.heatmap_max = max_val;
    }
}

int main() {
    // Setup window
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(1440, 900, "SpectraMap Desktop GUI (Native C++)", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Initialize Dear ImGui & ImPlot
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    apply_premium_style();
    
    AppState state;
    refresh_data_files(state);
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Sidebar control panel
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        int w_width, w_height;
        glfwGetWindowSize(window, &w_width, &w_height);
        ImGui::SetNextWindowSize(ImVec2(350, (float)w_height));
        
        ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
        
        ImGui::TextColored(ImVec4(0.56f, 0.45f, 0.90f, 1.00f), "SpectraMap C++ Backend");
        ImGui::Separator();
        ImGui::Spacing();
        
        // Section: Data Loading
        if (ImGui::CollapsingHeader("Data Loading", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (state.data_files.empty()) {
                ImGui::Text("No datasets found in data/");
            } else {
                std::vector<const char*> filenames;
                for (const auto& file : state.data_files) {
                    filenames.push_back(file.c_str());
                }
                ImGui::Combo("Select File", &state.selected_file_idx, filenames.data(), filenames.size());
            }
            
            const char* data_types[] = {"hyper_image", "multi_spectra", "single_spectrum"};
            static int type_idx = 0;
            ImGui::Combo("Data Type", &type_idx, data_types, 3);
            state.selected_data_type = data_types[type_idx];
            
            if (ImGui::Button("Load Dataset", ImVec2(-1, 30))) {
                state.is_loading = true;
                state.load_error = "";
                std::string path = "data/" + state.data_files[state.selected_file_idx];
                
                if (spmap::load_dataset(path, state.selected_data_type, state.current_dataset)) {
                    state.original_dataset = state.current_dataset;
                    state.is_loaded = true;
                    state.load_error = "";
                    
                    // Reset visualization values
                    state.selected_wavenumber_idx = state.current_dataset.wavenumbers.size() / 2;
                    state.selected_wavenumber = state.current_dataset.wavenumbers[state.selected_wavenumber_idx];
                    state.keep_min = state.current_dataset.wavenumbers.front();
                    state.keep_max = state.current_dataset.wavenumbers.back();
                    state.selected_pixel_idx = -1;
                    
                    // Reset analysis state
                    state.has_pca = false;
                    state.has_kmeans = false;
                    state.has_unmixing = false;
                    
                    update_heatmap(state);
                } else {
                    state.load_error = "Could not load dataset. Make sure dependencies (python) are active.";
                    state.is_loaded = false;
                }
                state.is_loading = false;
            }
            
            if (!state.load_error.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s", state.load_error.c_str());
            }
            
            if (state.is_loaded) {
                ImGui::Text("Loaded: %s", state.data_files[state.selected_file_idx].c_str());
                ImGui::Text("Shape: %d pixels, %d bands", (int)state.current_dataset.data.rows(), (int)state.current_dataset.data.cols());
                ImGui::Text("Grid: %d x %d x %d", state.current_dataset.m_grid, state.current_dataset.n_grid, state.current_dataset.l_grid);
            }
        }
        
        ImGui::Spacing();
        
        // Section: Preprocessing (only if loaded)
        if (state.is_loaded) {
            if (ImGui::CollapsingHeader("Preprocessing", ImGuiTreeNodeFlags_DefaultOpen)) {
                // Keep Range
                ImGui::DragFloatRange2("Keep Wavenumber", &state.keep_min, &state.keep_max, 1.0f, 
                                     (float)state.original_dataset.wavenumbers.front(), 
                                     (float)state.original_dataset.wavenumbers.back());
                if (ImGui::Button("Apply Keep", ImVec2(-1, 24))) {
                    std::vector<int> keep_indices;
                    for (size_t i = 0; i < state.current_dataset.wavenumbers.size(); ++i) {
                        double w = state.current_dataset.wavenumbers[i];
                        if (w >= state.keep_min && w <= state.keep_max) {
                            keep_indices.push_back(i);
                        }
                    }
                    if (!keep_indices.empty()) {
                        std::vector<double> new_wavenumbers;
                        Eigen::MatrixXd new_data(state.current_dataset.data.rows(), keep_indices.size());
                        for (size_t j = 0; j < keep_indices.size(); ++j) {
                            new_wavenumbers.push_back(state.current_dataset.wavenumbers[keep_indices[j]]);
                            new_data.col(j) = state.current_dataset.data.col(keep_indices[j]);
                        }
                        state.current_dataset.wavenumbers = new_wavenumbers;
                        state.current_dataset.data = new_data;
                        state.selected_wavenumber_idx = new_wavenumbers.size() / 2;
                        state.selected_wavenumber = new_wavenumbers[state.selected_wavenumber_idx];
                        update_heatmap(state);
                    }
                }
                
                ImGui::Separator();
                
                // SNIP
                ImGui::SliderInt("SNIP Iters", &state.snip_iter, 1, 100);
                if (ImGui::Button("Apply SNIP", ImVec2(-1, 24))) {
                    Eigen::MatrixXd baselines;
                    spmap::snip(state.current_dataset.data, state.snip_iter, baselines);
                    state.current_dataset.data -= baselines;
                    update_heatmap(state);
                }
                
                ImGui::Separator();
                
                // Gaussian
                ImGui::SliderFloat("Gaussian Sigma", &state.gaussian_sigma, 0.5f, 10.0f, "%.1f");
                if (ImGui::Button("Apply Gaussian", ImVec2(-1, 24))) {
                    Eigen::MatrixXd smoothed;
                    spmap::gaussian_smooth(state.current_dataset.data, state.gaussian_sigma, smoothed);
                    state.current_dataset.data = smoothed;
                    update_heatmap(state);
                }
                
                ImGui::Separator();
                
                // Vector Norm
                if (ImGui::Button("Apply L2 Vector Normalization", ImVec2(-1, 24))) {
                    Eigen::MatrixXd normed;
                    spmap::vector_normalize(state.current_dataset.data, normed);
                    state.current_dataset.data = normed;
                    update_heatmap(state);
                }
                
                ImGui::Separator();
                
                // Spikes
                ImGui::SliderInt("Spike Win", &state.spike_size, 1, 10);
                ImGui::SliderFloat("Spike Limit", &state.spike_limit, 2.0f, 15.0f, "%.1f");
                if (ImGui::Button("Apply Spike Removal", ImVec2(-1, 24))) {
                    Eigen::MatrixXd fixed;
                    spmap::fix_spikes(state.current_dataset.data, state.spike_size, state.spike_limit, fixed);
                    state.current_dataset.data = fixed;
                    update_heatmap(state);
                }
                
                ImGui::Separator();
                ImGui::Spacing();
                
                if (ImGui::Button("Reset to Raw Data", ImVec2(-1, 26))) {
                    state.current_dataset = state.original_dataset;
                    state.selected_wavenumber_idx = state.current_dataset.wavenumbers.size() / 2;
                    state.selected_wavenumber = state.current_dataset.wavenumbers[state.selected_wavenumber_idx];
                    state.selected_pixel_idx = -1;
                    state.has_pca = false;
                    state.has_kmeans = false;
                    state.has_unmixing = false;
                    update_heatmap(state);
                }
            }
            
            ImGui::Spacing();
            
            // Section: Analysis
            if (ImGui::CollapsingHeader("Analysis & Unmixing", ImGuiTreeNodeFlags_DefaultOpen)) {
                const char* analyses[] = {"None", "PCA", "K-Means Clustering", "Linear Unmixing"};
                ImGui::Combo("Analysis Mode", &state.analysis_type, analyses, 4);
                
                if (state.analysis_type == 1) { // PCA
                    ImGui::SliderInt("PCA Comps", &state.pca_components, 2, 5);
                    if (ImGui::Button("Run PCA", ImVec2(-1, 24))) {
                        spmap::run_pca(state.current_dataset.data, state.pca_components, state.pca_scores, state.pca_loadings);
                        state.has_pca = true;
                        state.selected_pca_component = 0;
                        update_heatmap(state);
                    }
                } 
                else if (state.analysis_type == 2) { // K-Means
                    ImGui::SliderInt("Clusters K", &state.kmeans_k, 2, 8);
                    if (ImGui::Button("Run K-Means", ImVec2(-1, 24))) {
                        spmap::run_kmeans(state.current_dataset.data, state.kmeans_k, state.kmeans_labels, state.kmeans_centroids);
                        state.has_kmeans = true;
                        update_heatmap(state);
                    }
                } 
                else if (state.analysis_type == 3) { // Linear Unmixing
                    const char* constraints[] = {"NNLS (Non-negative)", "OLS (Unconstrained)"};
                    ImGui::Combo("Constraint", &state.unmixing_constraint, constraints, 2);
                    
                    if (!state.has_kmeans) {
                        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Warning: Run K-Means first\nto define Endmembers!");
                    } else {
                        if (ImGui::Button("Run Abundance Unmixing", ImVec2(-1, 24))) {
                            if (state.unmixing_constraint == 0) {
                                spmap::run_nnls(state.current_dataset.data, state.kmeans_centroids, 500, state.abundances);
                            } else {
                                spmap::run_ols(state.current_dataset.data, state.kmeans_centroids, state.abundances);
                            }
                            state.has_unmixing = true;
                            state.selected_abundance_component = 0;
                            update_heatmap(state);
                        }
                    }
                }
            }
        }
        
        ImGui::End();
        
        // Main view area
        ImGui::SetNextWindowPos(ImVec2(350, 0));
        ImGui::SetNextWindowSize(ImVec2((float)w_width - 350, (float)w_height));
        ImGui::Begin("Visualization Window", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
        
        if (!state.is_loaded) {
            ImGui::SetCursorPos(ImVec2(((float)w_width - 350) / 2.0f - 100.0f, (float)w_height / 2.0f - 20.0f));
            ImGui::Text("Please load a dataset to begin visualization.");
        } else {
            ImGui::BeginTabBar("Main Tabs");
            
            // Tab 1: Spectral Mapping
            if (ImGui::BeginTabItem("Spectral Mapping")) {
                // Layout: Left (Spectral Plot), Right (Spatial Map)
                ImVec2 content_sz = ImGui::GetContentRegionAvail();
                
                // Let's draw the spectral line plot (top or left)
                ImGui::BeginChild("Spectral Child", ImVec2(content_sz.x * 0.55f, content_sz.y), true);
                ImGui::Text("Spectral Profiles");
                
                if (ImPlot::BeginPlot("Wavenumber Plot", ImVec2(-1, -1))) {
                    ImPlot::SetupAxes("Wavenumber (cm-1)", "Intensity", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
                    
                    int m_bands = state.current_dataset.wavenumbers.size();
                    
                    // 1. Plot Average Spectrum
                    std::vector<double> avg_spectrum(m_bands, 0.0);
                    int n_pixels = state.current_dataset.data.rows();
                    for (int j = 0; j < m_bands; ++j) {
                        for (int i = 0; i < n_pixels; ++i) {
                            avg_spectrum[j] += state.current_dataset.data(i, j);
                        }
                        avg_spectrum[j] /= n_pixels;
                    }
                    ImPlot::PlotLine("Average Spectrum", state.current_dataset.wavenumbers.data(), avg_spectrum.data(), m_bands);
                    
                    // 2. Plot Centroids if K-Means is run
                    if (state.has_kmeans) {
                        for (int c = 0; c < state.kmeans_k; ++c) {
                            std::string label = "Centroid " + std::to_string(c);
                            std::vector<double> cent_data(m_bands);
                            for (int j = 0; j < m_bands; ++j) {
                                cent_data[j] = state.kmeans_centroids(c, j);
                            }
                            ImPlot::PlotLine(label.c_str(), state.current_dataset.wavenumbers.data(), cent_data.data(), m_bands);
                        }
                    }
                    
                    // 3. Plot Selected Pixel Spectrum if selected
                    if (state.selected_pixel_idx >= 0 && state.selected_pixel_idx < n_pixels) {
                        std::string label = "Pixel " + std::to_string(state.selected_pixel_idx);
                        std::vector<double> pixel_data(m_bands);
                        for (int j = 0; j < m_bands; ++j) {
                            pixel_data[j] = state.current_dataset.data(state.selected_pixel_idx, j);
                        }
                        ImPlot::PlotLine(label.c_str(), state.current_dataset.wavenumbers.data(), pixel_data.data(), m_bands);
                    }
                    
                    // 4. Vertical draggable line for wavenumber selection
                    double drag_v = state.selected_wavenumber;
                    if (ImPlot::DragLineX(1234, &drag_v, ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f)) {
                        // Clamp
                        if (drag_v < state.current_dataset.wavenumbers.front()) drag_v = state.current_dataset.wavenumbers.front();
                        if (drag_v > state.current_dataset.wavenumbers.back()) drag_v = state.current_dataset.wavenumbers.back();
                        state.selected_wavenumber = drag_v;
                        
                        // Find closest index
                        auto it = std::lower_bound(state.current_dataset.wavenumbers.begin(), state.current_dataset.wavenumbers.end(), drag_v);
                        state.selected_wavenumber_idx = std::distance(state.current_dataset.wavenumbers.begin(), it);
                        update_heatmap(state);
                    }
                    
                    ImPlot::EndPlot();
                }
                ImGui::EndChild();
                
                ImGui::SameLine();
                
                // Right panel: Spatial Heatmap
                ImGui::BeginChild("Spatial Child", ImVec2(-1, -1), true);
                
                // Controls for Heatmap
                if (state.analysis_type == 2 && state.has_kmeans) {
                    ImGui::Text("Visualizing: K-Means Clusters");
                } 
                else if (state.analysis_type == 3 && state.has_unmixing) {
                    ImGui::Text("Visualizing: Component Abundance");
                    std::vector<const char*> components;
                    for (int c = 0; c < state.kmeans_k; ++c) {
                        components.push_back(nullptr); // will display as index
                    }
                    static int comp_sel = 0;
                    if (ImGui::SliderInt("Component", &comp_sel, 0, state.kmeans_k - 1)) {
                        state.selected_abundance_component = comp_sel;
                        update_heatmap(state);
                    }
                } 
                else if (state.analysis_type == 1 && state.has_pca) {
                    ImGui::Text("Visualizing: PCA Scores");
                    static int pca_sel = 0;
                    if (ImGui::SliderInt("PC Index", &pca_sel, 0, state.pca_components - 1)) {
                        state.selected_pca_component = pca_sel;
                        update_heatmap(state);
                    }
                } 
                else {
                    ImGui::Text("Visualizing: Wavenumber Intensity");
                    ImGui::Text("Current Wavenumber: %.1f cm-1", state.selected_wavenumber);
                }
                
                int m_grid = state.current_dataset.m_grid;
                int n_grid = state.current_dataset.n_grid;
                
                if (ImPlot::BeginPlot("Spatial Map View", ImVec2(-1, -1), ImPlotFlags_Equal)) {
                    ImPlot::SetupAxes("X Pixel", "Y Pixel", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
                    ImPlot::SetupAxesLimits(0, m_grid, 0, n_grid);
                    
                    // Colormap
                    ImPlotColormap cmap = ImPlotColormap_Viridis;
                    if (state.analysis_type == 2 && state.has_kmeans) {
                        cmap = ImPlotColormap_Deep;
                    }
                    ImPlot::PushColormap(cmap);
                    
                    ImPlot::PlotHeatmap("Map", state.heatmap_data.data(), n_grid, m_grid, 
                                      state.heatmap_min, state.heatmap_max, "%.1f", 
                                      ImPlotPoint(0, 0), ImPlotPoint(m_grid, n_grid));
                                      
                    ImPlot::PopColormap();
                    
                    // Handle clicking to select a pixel
                    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0)) {
                        ImPlotPoint pt = ImPlot::GetPlotMousePos();
                        int click_x = static_cast<int>(std::floor(pt.x));
                        int click_y = static_cast<int>(std::floor(pt.y));
                        
                        if (click_x >= 0 && click_x < m_grid && click_y >= 0 && click_y < n_grid) {
                            // Find the pixel index corresponding to these coords
                            int n_pixels = state.current_dataset.data.rows();
                            for (int i = 0; i < n_pixels; ++i) {
                                int px = static_cast<int>(state.current_dataset.position(i, 0));
                                int py = static_cast<int>(state.current_dataset.position(i, 1));
                                if (px == click_x && py == click_y) {
                                    state.selected_pixel_idx = i;
                                    break;
                                }
                            }
                        }
                    }
                    
                    ImPlot::EndPlot();
                }
                ImGui::EndChild();
                
                ImGui::EndTabItem();
            }
            
            // Tab 2: PCA Scores Scatter
            if (state.has_pca && ImGui::BeginTabItem("PCA Scores")) {
                ImGui::Text("PCA Scores Scatter (PC1 vs PC2)");
                
                if (ImPlot::BeginPlot("PCA Scatter", ImVec2(-1, -1))) {
                    ImPlot::SetupAxes("PC1 Score", "PC2 Score", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
                    
                    int n_pixels = state.current_dataset.data.rows();
                    std::vector<double> pc1(n_pixels), pc2(n_pixels);
                    for (int i = 0; i < n_pixels; ++i) {
                        pc1[i] = state.pca_scores(i, 0);
                        pc2[i] = state.pca_scores(i, 1);
                    }
                    
                    if (state.has_kmeans) {
                        // Plot colored by cluster
                        ImPlot::PushColormap(ImPlotColormap_Deep);
                        for (int c = 0; c < state.kmeans_k; ++c) {
                            std::vector<double> c_pc1, c_pc2;
                            for (int i = 0; i < n_pixels; ++i) {
                                if (state.kmeans_labels[i] == c) {
                                    c_pc1.push_back(pc1[i]);
                                    c_pc2.push_back(pc2[i]);
                                }
                            }
                            std::string label = "Cluster " + std::to_string(c);
                            if (!c_pc1.empty()) {
                                ImPlot::PlotScatter(label.c_str(), c_pc1.data(), c_pc2.data(), c_pc1.size());
                            }
                        }
                        ImPlot::PopColormap();
                    } else {
                        ImPlot::PlotScatter("Pixels", pc1.data(), pc2.data(), n_pixels);
                    }
                    
                    ImPlot::EndPlot();
                }
                ImGui::EndTabItem();
            }
            
            // Tab 3: Endmember Abundances
            if (state.has_unmixing && ImGui::BeginTabItem("Abundance Maps")) {
                ImGui::Text("Abundance maps for all endmembers");
                
                int m_grid = state.current_dataset.m_grid;
                int n_grid = state.current_dataset.n_grid;
                int n_pixels = state.current_dataset.data.rows();
                
                // Show side-by-side or select from grid
                ImVec2 avail = ImGui::GetContentRegionAvail();
                float panel_w = std::max(200.0f, avail.x / state.kmeans_k - 10.0f);
                
                for (int c = 0; c < state.kmeans_k; ++c) {
                    ImGui::BeginChild(("AbundChild_" + std::to_string(c)).c_str(), ImVec2(panel_w, avail.y), true);
                    ImGui::Text("Component %d", c);
                    
                    std::vector<double> ab_data(m_grid * n_grid, 0.0);
                    for (int i = 0; i < n_pixels; ++i) {
                        int x = static_cast<int>(state.current_dataset.position(i, 0));
                        int y = static_cast<int>(state.current_dataset.position(i, 1));
                        if (x >= 0 && x < m_grid && y >= 0 && y < n_grid) {
                            ab_data[y * m_grid + x] = state.abundances(i, c);
                        }
                    }
                    
                    if (ImPlot::BeginPlot(("AbundPlot_" + std::to_string(c)).c_str(), ImVec2(-1, -1), ImPlotFlags_Equal)) {
                        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
                        ImPlot::PushColormap(ImPlotColormap_Viridis);
                        ImPlot::PlotHeatmap("Abundance", ab_data.data(), n_grid, m_grid, 0.0, 1.0, nullptr,
                                          ImPlotPoint(0, 0), ImPlotPoint(m_grid, n_grid));
                        ImPlot::PopColormap();
                        ImPlot::EndPlot();
                    }
                    ImGui::EndChild();
                    if (c < state.kmeans_k - 1) ImGui::SameLine();
                }
                ImGui::EndTabItem();
            }
            
            ImGui::EndTabBar();
        }
        ImGui::End();
        
        // Render
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.06f, 0.06f, 0.08f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
