# Data collapse summary (for collaborators)

## Data input
- Please place your raw data under a directory named `Data_scale/`.
- The script will scan files, infer L from filenames or parent folders (e.g., `L=9_*.csv`, `.../L=11/data.csv`).
- Each file should provide columns like: `U, R01, sigma` (or `U, Y, sigma`). Column names are auto-mapped.

## Ansatz
- Data collapse (drop L=7):  Y ≈ f((U − U_c) L^{1/ν})
- Data collapse (all L, with finite size correction):  Y ≈ f((U − U_c) L^{1/ν}) × (1 + b L^c)
- Plotting for finite-size correction: normalized by s_norm=(1+b L^c)/(1+b L_ref^c), L_ref=geom mean

## Recommended parameters (median ± uncertainty)
- Data collapse (drop L=7):  U_c = 8.669546 ± 0.002214,   ν^(-1) = 1.191639 ± 0.011695
- Data collapse (all L, with finite size correction):  U_c = 8.448503 ± 0.074273,   ν^(-1) = 1.285148 ± 0.065871
- (b,c) plotting medians: (0.797789, -1.063099)

## One-click reproduce (fit + plots)
1. pip install -r requirements.txt
2. python make_collapse.py --data-scale Data_scale
   - The script will: build `real_data_combined.csv` from Data_scale → fit via datacollapse → generate three figures.
   - Update `Data_scale/` (e.g., add new L), then re-run to refresh all results.
