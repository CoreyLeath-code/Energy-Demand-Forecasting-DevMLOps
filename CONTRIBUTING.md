# Contributing to Energy Demand Forecasting DevMLOps

Thank you for contributing. Changes should preserve correctness, reproducibility, security, observability, and deployment clarity across the training, API, Streamlit, container, and infrastructure layers.

## Development setup

```bash
git clone https://github.com/CoreyLeath-code/Energy-Demand-Forecasting-DevMLOps.git
cd Energy-Demand-Forecasting-DevMLOps
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the full training and MLOps stack only when working on deep-learning or experiment-tracking features:

```bash
pip install -r requirements.txt
```

## Required local validation

```bash
ruff check src/serve src/forecasting.py tests/test_api.py tests/test_forecasting.py streamlit_app.py streamlit_demo/app.py \
  --select E9,F63,F7,F82

python -m compileall -q src tests streamlit_app.py streamlit_demo/app.py

pytest tests/test_api.py tests/test_forecasting.py -v \
  --cov=src.serve \
  --cov=src.forecasting \
  --cov-report=term-missing

docker build -t energy-forecasting:local .
```

Validate the public dashboard locally with:

```bash
pip install -r streamlit_demo/requirements.txt
streamlit run streamlit_demo/app.py
```

## Branch and commit conventions

Use focused branches such as:

```text
feat/forecast-confidence-bands
fix/api-model-loading
refactor/feature-validation
docs/streamlit-deployment
```

Use Conventional Commits when practical:

```text
feat: add temperature-aware forecast features
fix: preserve request-body validation
refactor: isolate model artifact loading
test: cover missing-data failure modes
ci: add Streamlit deployment smoke test
docs: explain rollback procedure
```

## Pull-request standard

A pull request should explain:

- the problem and user or operational impact;
- the chosen design and relevant tradeoffs;
- test and deployment evidence;
- data, model, API, or infrastructure compatibility impact;
- observability and security considerations;
- rollback or recovery steps.

Add or update tests for behavior changes. Avoid mixing unrelated refactors and features in the same pull request.

## Data and model safety

Do not commit:

- private or licensed datasets;
- customer or production energy records;
- credentials, tokens, `.env` files, or Streamlit secrets;
- large model checkpoints or experiment artifacts;
- generated MLflow stores, raw telemetry, or local state.

Use synthetic or properly licensed public data for examples. Public demonstrations must clearly identify synthetic data and baseline-model output.

## Review criteria

Reviewers should assess:

- correctness and failure behavior;
- time-series leakage and reproducibility;
- request and dataset validation;
- model and feature compatibility;
- performance and resource use;
- security and supply-chain impact;
- observability and operational readiness;
- deployment and rollback safety;
- documentation accuracy.

## Release expectations

Release candidates should have green multi-version tests, Streamlit smoke tests, container health validation, security workflows, and release-readiness checks. Tag releases using Semantic Versioning in the form `vMAJOR.MINOR.PATCH`.
