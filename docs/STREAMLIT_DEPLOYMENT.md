# Streamlit Community Cloud Deployment

Energy Demand Forecasting DevMLOps includes a lightweight, artifact-independent public dashboard designed for Streamlit Community Cloud.

## Deployment settings

After the pull request is merged, create an app with:

```text
Repository:
CoreyLeath-code/Energy-Demand-Forecasting-DevMLOps

Branch:
main

Main file path:
streamlit_demo/app.py
```

The app does not require API keys, private datasets, model checkpoints, scalers, or external services for its built-in demonstration.

## Why the app lives in `streamlit_demo/`

Streamlit Community Cloud installs the dependency manifest closest to the selected entry point. The dedicated directory keeps the public deployment small and avoids installing the full PyTorch, MLflow, XGBoost, training, and infrastructure toolchain.

Deployment files:

- `streamlit_demo/app.py` — Community Cloud entry point.
- `streamlit_demo/requirements.txt` — lightweight dashboard dependencies.
- `.python-version` — Python 3.11 runtime pin.
- `.streamlit/config.toml` — headless server and theme configuration.
- `streamlit_app.py` — dashboard implementation shared by local and cloud execution.

## Local validation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r streamlit_demo/requirements.txt
streamlit run streamlit_demo/app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit server health endpoint is:

```text
http://localhost:8501/_stcore/health
```

## Public-demo behavior

The dashboard supports two explicit data sources:

1. **Synthetic demo data** — reproducible hourly energy demand generated from a fixed seed, daily and weekly seasonality, weather effects, trend, and noise.
2. **Uploaded CSV** — user-provided data with selectable timestamp and demand columns.

The public forecast uses a deterministic seasonal-naive baseline with bounded trend adjustment and uncertainty intervals. The application identifies this backend clearly and does not claim that synthetic output is a production forecast.

The full repository also contains LSTM, GRU, Transformer, model-training, FastAPI, Docker, Kubernetes, Helm, MLflow, and monitoring pathways. Those heavier components are intentionally excluded from the public Streamlit dependency set.

## Uploaded CSV contract

The uploaded file must contain:

- one timestamp-like column;
- one non-negative numeric demand column;
- at least 48 valid observations;
- enough history for two selected seasonal periods.

The dashboard:

- parses timestamps;
- coerces demand values to numeric;
- removes invalid rows;
- rejects negative demand;
- aggregates duplicate timestamps using the mean;
- sorts observations chronologically;
- reports validation results.

## Secrets

No secrets are required for the built-in app.

When optional external services are added later, configure credentials through Streamlit Community Cloud's encrypted secrets interface. Never commit `.streamlit/secrets.toml`, `.env` files, credentials, private data, or production endpoints.

## CI validation

The Enterprise CI workflow:

1. installs only `streamlit_demo/requirements.txt`;
2. imports the dashboard module;
3. executes a deterministic forecast contract;
4. starts Streamlit in headless mode;
5. verifies `/_stcore/health`;
6. requires the Streamlit job to pass before release readiness can pass.

## Troubleshooting

### Dependency installation is slow

Confirm the main file path is exactly:

```text
streamlit_demo/app.py
```

Selecting the root `streamlit_app.py` may cause Streamlit Cloud to use the full root requirements file, which includes the training and MLOps stack.

### The app reports insufficient history

Upload more observations or select a shorter seasonal period. Forecasting requires at least two full seasonal cycles.

### The uploaded CSV is rejected

Verify that the selected timestamp column can be parsed as dates and that the selected demand column contains non-negative numeric values.

### The app becomes inactive

Streamlit Community Cloud may sleep inactive community applications. Open the app and allow the platform to wake or reboot it from the app management console.
