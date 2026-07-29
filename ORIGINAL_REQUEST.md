# Original User Request

## Initial Request — 2026-07-22T08:43:52Z

Overhaul the SpectraMap Raman analysis web application to make it robust, interactive, minimalist, and low-friction, featuring an automatic execution workflow upon dataset loading.

Working directory: c:\Users\Juan\Documents\GitHub\spectramap
Integrity mode: development

## Requirements

### R1. Automatic Execution & Minimalist UX
Automatically trigger the end-to-end Raman analysis pipeline using intelligent default parameters as soon as a dataset is loaded or selected. Clean up the UI sidebar using clean collapsible expanders to minimize visual noise and manual user clicks.

### R2. Application Robustness & Graceful Fallbacks
Enforce strict input validation and graceful fallback handling across file loading, cosmic ray removal, glass subtraction, baseline correction, VCA unmixing, HCA clustering, and PCA. Ensure no unhandled tracebacks or UI crashes occur even on malformed or edge-case datasets.

### R3. Seamless Saving & Exporting
Maintain full automated saving of all generated figures, preprocessed data CSVs, score/loading tables, and metadata JSON files to the output directory upon pipeline completion, ensuring minimal effort to reach final output files.

## Acceptance Criteria

### Automation & Performance
- [ ] Loading or selecting a dataset automatically triggers analysis execution using smart defaults without requiring manual button clicks
- [ ] Pipeline results are generated and rendered in clean tabs (Maps, Loadings/Spectra, Scatter/Stack, Data) within seconds of dataset load

### Robustness & User Experience
- [ ] Malformed or missing optional inputs (e.g. missing glass spectrum files, zero-power laser settings, single-cluster edge cases) are handled with informative UI banners instead of app tracebacks
- [ ] All generated figures (score maps grid, stacked loadings, correlation heatmaps, abundance maps) render cleanly without clipped labels
- [ ] Static compilation `python -m py_compile app.py` passes cleanly

## Follow-up — 2026-07-22T08:51:09Z

Enhance the SpectraMap Raman analysis GUI to be ultra-robust, highly interactive, minimalist, and visually clean, minimizing user friction to reach final analytical results and publication-quality exports.

Working directory: c:\Users\Juan\Documents\GitHub\spectramap
Integrity mode: development

## Requirements

### R1. Minimalist UI & Intuitive Interactivity
Streamline the Streamlit user interface (`app.py`) to reduce visual clutter, group parameters cleanly into collapsible expanders, and provide immediate, live feedback across interactive plots (spatial map rotation/flipping/cropping, spectral zoom, loadings stack, scatter plots, and cluster averages).

### R2. End-to-End Application Robustness
Ensure rock-solid stability across dataset loading, automatic execution, baseline subtraction, cosmic ray removal, unmixing (VCA), principal component analysis (PCA), hierarchical clustering (HCA), and HDBSCAN. All edge cases (e.g. zero variance, missing coordinates, custom file formats, missing glass files) must handle gracefully with clear UI notices instead of tracebacks.

### R3. Seamless High-Resolution Multi-Format Export
Maintain automated, one-click saving of preprocessed spectra, score/loading/label CSV tables, metadata JSON files, and multi-format publication figures (`.png`, `.pdf`, `.svg`) to the output directory upon pipeline completion.

## Acceptance Criteria

### Automation & Stability
- [ ] Loading or selecting any dataset automatically triggers full analysis using intelligent laser preset defaults without crashing
- [ ] 2D spatial orientation (rotation/flipping) and spatial cropping work seamlessly across VCA, PCA, HCA, and HDBSCAN map renderers
- [ ] All 84+ unit test assertions pass clean (`pytest`)
- [ ] Static compilation `python -m py_compile app.py tools/witec_raman_pipeline.py src/spectramap/spmap.py` completes cleanly without errors

## Follow-up — 2026-07-27T10:57:28Z

Reorganize and optimize the SpectraMap Raman analysis web interface (`app.py`) into an intuitive, step-by-step workflow that guides users logically from dataset loading to spatial mapping, reference spectra inspection, quantification, and data export.

Working directory: c:\Users\Juan\Documents\GitHub\spectramap
Integrity mode: development

## Requirements

### R1. Step-by-Step Guided UI Flow
Structure the main interface and sidebar into a clear, numbered 5-step sequential analytical workflow:
- **Step 1: Data Input & Selection**: File upload, sample selection, or folder loading with automatic dataset parsing.
- **Step 2: Spatial Mapping & Co-localization**: Endmember abundance maps, 2D orientation/cropping, colormap selection, and 3/4-color co-localization overlay maps.
- **Step 3: Reference Spectra & Peak Analysis**: Overlapped & individual endmember reference spectra, peak finding, and Y-axis offset controls.
- **Step 4: Quantification & Downstream Statistics**: Pearson correlation heatmap, biochemical peak ratios, and PCA score scatter plots synchronized around chosen endmembers.
- **Step 5: Export & Data Table Inspection**: Preprocessed data CSV previews, figure export buttons, and run metadata.

### R2. Global Endmember Selection Synchronization
Ensure endmember selections made at any step (e.g. Step 2 or Step 3) seamlessly filter all downstream grids, overlays, correlation matrices, and quantification bar charts across all tabs.

### R3. Visual Clarity & Minimal User Friction
Maintain clean layout organization with step indicators, intuitive control placement, colorblind-friendly overlay options, and automated execution on dataset load.

## Acceptance Criteria

### Workflow & Interactivity
- [ ] Interface clearly presents a numbered 5-step workflow (Step 1 through Step 5) guiding the user sequentially through the analytical process
- [ ] Endmember selections made in Step 2 or 3 automatically update all downstream maps, grids, heatmap matrices, and ratio bar charts
- [ ] 3-channel and 4-channel co-localization overlay maps render cleanly with colorblind-friendly palette modes
- [ ] Static compilation `python -m py_compile app.py tools/witec_raman_pipeline.py src/spectramap/spmap.py` passes cleanly without errors

## Follow-up — 2026-07-29T13:51:56Z

Perform a comprehensive end-to-end audit, automated feature test run, and stability verification of the SpectraMap Raman analysis application (`app.py`, `witec_raman_pipeline.py`, `spmap.py`) across single-dataset and multi-dataset batch modes.

Working directory: c:\Users\Juan\Documents\GitHub\spectramap
Integrity mode: development

## Requirements

### R1. Comprehensive Automated Unit & Pipeline Verification
Execute the full pytest suite and automated pipeline verification scripts covering single-sample datasets, multi-file batch datasets, wavenumber resampling, PCA, VCA unmixing, HCA clustering, HDBSCAN, and figure/CSV exports.

### R2. End-to-End GUI & Web Application Stability Audit
Verify that `app.py` compiles cleanly, runs without unhandled tracebacks or Streamlit exceptions, handles edge cases (e.g. empty inputs, zero variance, single cluster) gracefully with informative UI banners, and renders all 5 steps properly.

### R3. Export Artifact Integrity Check
Confirm that all generated figures (`.png`, `.pdf`, `.svg`), preprocessed spectra CSVs, PCA/VCA score and loading matrices, and pipeline JSON metadata files write cleanly to output directories without missing fields.

## Acceptance Criteria

### Verification & Robustness
- [ ] Pytest test suite (`pytest`) runs with 100% passing status across all 105+ test items
- [ ] Static compilation `python -m py_compile app.py tools/witec_raman_pipeline.py src/spectramap/spmap.py smart_importer.py test_smart_importer.py` passes 100% cleanly
- [ ] Multi-sample batch processing pipeline correctly resamples, stacks, and classifies datasets without errors
- [ ] All 5 workflow steps in the Streamlit UI render all interactive plots, heatmaps, and tables cleanly without tracebacks


## Follow-up — 2026-07-29T13:02:05Z

Verify, execute, and validate the hosted SpectraMap Raman analysis web application (`app.py`, `tools/witec_raman_pipeline.py`, `src/spectramap/spmap.py`) across all 5 workflow steps, multi-sample batch modes, and normalization features.

Working directory: c:\Users\Juan\Documents\GitHub\spectramap
Integrity mode: development

## Requirements

### R1. Full Pipeline & Web App Execution Audit
Ensure the Streamlit application starts and runs continuously on `http://localhost:8501`, executing all preprocessing, band position normalization, VCA unmixing, PCA, HCA clustering, and HDBSCAN without UI errors.

### R2. Multi-Dataset Batch & Classification Verification
Validate that multi-file batch datasets load cleanly, resample onto common wavenumber grids, compute group difference spectra, and generate sample classification composition matrices.

### R3. Automated Test Suite Integrity
Confirm all 105+ unit tests pass via `pytest` and static compilation (`py_compile`) succeeds with 0 errors across all codebase modules.

## Acceptance Criteria

### Execution & Stability
- [ ] Streamlit web server runs actively on port 8501 without process crashes
- [ ] Pytest test suite (`pytest`) runs with 100% passing status (0 failures)
- [ ] Static compilation `python -m py_compile app.py tools/witec_raman_pipeline.py src/spectramap/spmap.py` completes cleanly
- [ ] All 5 workflow steps render all interactive plots, heatmaps, and scatter plots without tracebacks




