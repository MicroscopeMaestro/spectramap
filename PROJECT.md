# Project: SpectraMap

## Architecture
- Streamlit application (`app.py`) interfacing with processing modules.
- Input: Raman dataset files (CSV, TXT, WDF, etc.), optional reference files.
- Pipeline steps: File loading -> Cosmic Ray Removal -> Glass Subtraction -> Baseline Correction -> VCA Unmixing -> HCA Clustering -> PCA.
- Output: Streamlit UI tabs (Maps, Loadings/Spectra, Scatter/Stack, Data) and auto-saved outputs (figures, CSVs, score/loading tables, metadata JSON).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1_UX_AutoExec | R1: Auto execution on load, smart defaults, collapsible expanders | none | DONE |
| 2 | M2_Robustness_Validation | R2: Input validation & graceful fallback banners across all steps | M1 | DONE |
| 3 | M3_AutoSave_Export | R3: Automated output saving of figures, CSVs, tables, metadata | M1, M2 | DONE |
| 4 | M4_Final_Integration_Verification | Quality: E2E tests, label rendering, static compilation, audit | M1, M2, M3 | DONE |
| 5 | M5_Followup_Verification | Follow-up: Interactive map ops (rot/flip/crop across VCA/PCA/HCA/HDBSCAN), 84+ unit tests, multi-module py_compile | M4 | DONE |
| 6 | M6_GuidedWorkflow_EndmemberSync | Follow-up 2026-07-27: Step-by-Step 5-Step Guided UI Flow, Global Endmember Sync, Colorblind-Friendly 3/4-channel overlays, py_compile & test pass | M5 | IN_PROGRESS |

## Interface Contracts
### UI ↔ Pipeline Execution
- Automatic execution triggered upon dataset upload or selection of sample dataset.
- State managed cleanly in `st.session_state`.
- Inputs validated before processing; warnings shown via `st.warning` / `st.error` without raising tracebacks.

## Code Layout
- `app.py`: Main Streamlit web application.
- Processing modules / utilities (if present in repo).
- Tests / test runner (to be set up/verified).
