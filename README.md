# icu24-calibrated-baselines

**Code-only, no patient data.** This repository provides reproducible, transparent ML baselines for first-24h ICU mortality prediction (MIMIC-IV schema, but runs with synthetic data only). No real patient data is ever included or committed.

## Quickstart

```bash
pip install -r requirements-pages.txt
python scripts/generate_synthetic.py
python scripts/run_baselines.py
python scripts/calibrate_and_eval.py
python scripts/make_figures.py
```

## License
MIT. See LICENSE file.

## Disclaimer
- No patient data is ever included or committed.
- Synthetic data only; to reproduce with MIMIC-IV, point pipeline at your credentialed tables.
- Never upload real data.
