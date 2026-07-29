# SpectraMap Test Infrastructure Specification (`TEST_INFRA.md`)

## 1. Executive Summary
This document defines the E2E testing framework and infrastructure for **SpectraMap**, a hyperspectral Raman spectroscopy analysis suite. It specifies the 4-tier testing methodology, feature inventory, test architecture, and coverage thresholds required for continuous integration and automated verification.

---

## 2. Feature Inventory

The SpectraMap software architecture is organized into seven primary feature categories:

| Feature ID | Feature Name | Description | Key Modules / Functions | Primary Inputs | Expected Outputs |
|---|---|---|---|---|---|
| **FEAT-01** | **Dataset Loading** | Multi-format loading (CSV, CSV.XZ, SPC, WITec TXT, MAT) & custom text import | `read_csv`, `read_csv_xz`, `read_csv_3d_xz`, `read_spc`, `load_witec_map`, `parse_with_ollama` | CSV, .csv.xz, .spc, .txt, .mat files | `hyper_object` / DataFrame with coordinates & spectra |
| **FEAT-02** | **Auto-Execution Pipeline** | End-to-end automated processing for 532 nm and 785 nm laser configurations | `witec_raman_pipeline.run`, preset configurations | Config dict, scan TXT, background TXT | Timestamped output dir with figures, CSVs, metadata |
| **FEAT-03** | **Pre-Processing** | Wavenumber cropping, SNIP/airPLS baseline correction, cosmic ray removal, glass subtraction, normalization | `keep`, `snip`, `airpls`, `cosmic_rays`, `glass_subtraction`, `vector`, `standard_normal_variate` | Spectral matrix, wavenumber array, reference spectrum | Cleaned & baseline-corrected spectral matrix |
| **FEAT-04** | **VCA Unmixing** | Vertex Component Analysis & Non-Negative Least Squares (NNLS) abundance estimation | `vca`, `VCA`, `NNLS`, `OLS` | Spectral matrix ($N \times P$), $K$ endmembers | Endmember spectra ($K \times P$), Abundance maps ($N \times K$) |
| **FEAT-05** | **HCA & Clustering** | Hierarchical Cluster Analysis, KMeans, and HDBSCAN density clustering | `hca`, `kmeans`, `hdbscan`, `plot_dendrogram` | Spectral or abundance matrix, cluster count | Cluster labels ($N$), Linkage matrix, distance dendrogram |
| **FEAT-06** | **PCA Analysis** | Principal Component Analysis fit, scores, loadings, explained variance | `pca`, `pca_fit`, `confidence_ellipse` | Spectral matrix, component count | PC scores ($N \times C$), Loadings ($C \times P$), Variance ratios |
| **FEAT-07** | **Output Saving & Export** | Automated generation and saving of CSV data tables, matplotlib figures, and metadata JSON | `save_data`, figure export routines in `witec_raman_pipeline` | Processed DataFrames, figures, config | CSV files, PNG figures, `metadata.json` schema |

---

## 3. 4-Tier Testing Methodology

The test suite is organized into four distinct tiers:

### Tier 1: Feature Coverage (Happy Path)
- **Objective**: Validate basic correctness of each feature under standard, expected operating conditions.
- **Scope**: At least 5 test cases per feature (35+ test cases total across FEAT-01 to FEAT-07).
- **Validation**: Confirm correct return types, expected output dimensions, mathematical invariants, and file creations.

### Tier 2: Boundary & Corner Cases
- **Objective**: Ensure robustness and graceful error/fallback behavior when facing invalid, extreme, or edge-case inputs.
- **Scope**: At least 5 test cases per feature (35+ test cases total).
- **Validation**: Cover missing glass reference files, zero laser power, single-cluster HCA ($K=1$), malformed CSV/TXT headers, NaN/Inf spectral values, out-of-range cropping, flat signal matrices, and read-only destination paths.

### Tier 3: Cross-Feature Interactions
- **Objective**: Verify seamless execution across pipeline multi-step sequences.
- **Scope**: At least 8 test cases covering feature interactions.
- **Validation**: Test interactions such as VCA + HCA, Glass Subtraction + airPLS + Normalization, Cosmic Ray Removal + SNIP + PCA, Smart Importer -> `hyper_object` -> VCA, and parallel vs. single-thread airPLS execution.

### Tier 4: Real-World Applications
- **Objective**: Execute realistic end-to-end dataset scenarios representing real analytical laboratory workflows.
- **Scope**: At least 6 test cases covering full-spectrum workflows.
- **Validation**: Test scenarios including simulated 532 nm WITec hyperspectral map, 785 nm fingerprint scan, real `3D.csv.xz` compressed dataset, real `paracetaminol.csv` peak calibration, real `messy_dataset.txt` custom parsing, and multi-polymer microplastics unmixing.

---

## 4. Test Architecture & Directory Structure

```
spectramap/
├── TEST_INFRA.md                # Infrastructure specification (this file)
├── TEST_READY.md                # Test execution summary & completion report
├── tests/                       # Test suite directory
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures (synthetic matrices, test files, configs)
│   ├── test_tier1_feature_coverage.py   # Tier 1 tests (35 test cases)
│   ├── test_tier2_boundary_corner.py     # Tier 2 tests (35 test cases)
│   ├── test_tier3_cross_feature.py      # Tier 3 tests (8 test cases)
│   ├── test_tier4_real_world.py         # Tier 4 tests (6 test cases)
│   └── run_all_tests.py                 # Automated test runner script
```

---

## 5. Coverage Thresholds & Quality Invariants

| Category | Requirement / Metric | Target Threshold |
|---|---|---|
| **Tier 1 Feature Coverage** | Test cases per feature (FEAT-01 to FEAT-07) | $\ge 5$ test cases per feature ($\ge 35$ total) |
| **Tier 2 Boundary Coverage** | Boundary & corner cases per feature | $\ge 5$ test cases per feature ($\ge 35$ total) |
| **Tier 3 Cross-Feature** | Interaction test cases | $\ge 8$ test cases |
| **Tier 4 Real-World** | End-to-end realistic application scenarios | $\ge 6$ test cases |
| **Overall Pass Rate** | Percentage of passing tests | **100%** (0 failures allowed) |
| **Code Integrity** | Genuine implementations only | No hardcoded assertions, dummy facade classes, or skipped tests |
| **Execution Time** | Total suite execution time | $< 60$ seconds |

---

## 6. Verification & Execution Commands

### Standard Pytest Execution
```bash
pytest -v tests/
```

### Direct Python Test Runner Execution
```bash
python tests/run_all_tests.py
```
