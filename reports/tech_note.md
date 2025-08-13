# Technical Note (Template)

## Methods
- Synthetic data only; see scripts/generate_synthetic.py
- Baselines: age-only LR, EBM, GBDT
- Calibration: Platt, isotonic
- Metrics: AUROC, AUPRC, Brier, ECE (bootstrap CI), decision curve
- Subgroup: sex, age bands
- TODO: Replace synthetic with local MIMIC path (never committed)
- TODO: Temporal split (train 2017–2019, test 2020–2022)

## Results (with CIs)
- [Insert results summary]

## Limitations
- Synthetic only; real data requires credentialed access
- No PHI; CPU-only; no persistence
