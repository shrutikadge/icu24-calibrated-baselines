# icu24-calibrated-baselines

## Project Summary
Transparent, reproducible ML baselines for first-24h ICU mortality (MIMIC-IV schema, synthetic demo only). Compare EBM vs GBDT, calibration, ECE, decision curve, and fairness slices.

## How to Reproduce Locally
- Install requirements: `pip install -r requirements-pages.txt`
- Generate synthetic data: `python scripts/generate_synthetic.py`
- Run baselines: `python scripts/run_baselines.py`
- Calibrate and evaluate: `python scripts/calibrate_and_eval.py`
- Make figures: `python scripts/make_figures.py`

## Figure Thumbnails
- See figures below (from `reports/figures/`)

## Data & License
> **Code-only, no patient data.** MIMIC-IV requires credentialed access. To reproduce with real data, point pipeline at your tables. Never upload data. MIT License.
