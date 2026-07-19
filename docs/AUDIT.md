# Reproducibility and Evidence Audit

## Controls implemented

- Seeded synthetic data generation and stable SHA-256 dataset fingerprints.
- Chronological holdout for XGBoost training; no shuffled time-series split.
- Explicit estimator seed, single-thread execution, and objective.
- Model metadata sidecar containing feature order, split boundaries, dependency versions, metrics, and SHA-256 hashes for input, config, and model artifacts.
- Expanding-window rolling-origin evaluation that predicts only observations after each training origin.
- MAE, RMSE, MAPE, sMAPE, MASE, bias, R², and fold-dispersion reporting.
- GitHub-hosted benchmark workflow with a JSON artifact and evidence contract.
- Python 3.10/3.11 CI with 23 tests and 93.10% measured coverage.

## Verification

Run locally:

```bash
PYTHONHASHSEED=0 python benchmarks/run_benchmark.py --iterations 500 --warmup 50 --output benchmark-results.json
pytest tests/test_api.py tests/test_forecasting.py tests/test_evaluation.py -v
```

Compare deterministic quality metrics and `data_sha256` with [`benchmarks/latest.json`](../benchmarks/latest.json). Latency may vary by hardware.

## Claim boundary

The committed benchmark is for the deterministic public baseline and seeded synthetic data. It does not validate the deep-learning architectures, a private trained artifact, real-grid generalization, calibrated probabilistic coverage, or production service-level objectives.
