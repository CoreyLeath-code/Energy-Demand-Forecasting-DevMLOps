"""Interactive Streamlit dashboard for Energy Demand Forecasting DevMLOps.

The public demo is artifact-independent: it can generate reproducible synthetic
energy data or accept a user-uploaded CSV, then run a transparent seasonal-naive
baseline. The trained PyTorch pipeline remains available elsewhere in the
repository for full model training and serving.
"""

from __future__ import annotations

import io
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.forecasting import (
    ForecastSettings,
    generate_demo_data,
    normalize_energy_frame,
    seasonal_naive_forecast,
    summarize_dataset,
)

REPOSITORY_URL = "https://github.com/CoreyLeath-code/Energy-Demand-Forecasting-DevMLOps"


@st.cache_data(show_spinner=False)
def cached_demo_data(days: int, seed: int) -> pd.DataFrame:
    """Generate a reusable synthetic dataset for a selected demo configuration."""

    return generate_demo_data(periods=days * 24, seed=seed)


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV and return an isolated DataFrame."""

    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("Uploaded CSV is empty.")
    return pd.read_csv(io.BytesIO(raw_bytes))


def render_sidebar() -> tuple[pd.DataFrame, int, int]:
    """Render data-source controls and return data plus forecast settings."""

    with st.sidebar:
        st.title("⚡ Energy Forecasting")
        st.caption("DevMLOps portfolio demonstration")

        source = st.radio(
            "Data source",
            options=("Synthetic demo", "Upload CSV"),
            help="The synthetic dataset is reproducible and contains no production data.",
        )

        if source == "Synthetic demo":
            days = st.slider("History window (days)", 14, 90, 30)
            seed = st.number_input("Synthetic seed", min_value=0, max_value=100_000, value=42)
            frame = cached_demo_data(days, int(seed))
            st.info("Using reproducible synthetic hourly demand data.")
        else:
            uploaded = st.file_uploader("Upload hourly energy CSV", type=("csv",))
            st.caption("Required logical fields: timestamp and demand.")
            if uploaded is None:
                st.warning("Upload a CSV to continue. Synthetic data is never substituted silently.")
                st.stop()

            raw = load_uploaded_csv(uploaded)
            if len(raw.columns) < 2:
                st.error("The uploaded CSV must contain at least two columns.")
                st.stop()

            timestamp_column = st.selectbox("Timestamp column", options=list(raw.columns), index=0)
            demand_candidates = [column for column in raw.columns if column != timestamp_column]
            demand_column = st.selectbox("Demand column", options=demand_candidates, index=0)
            frame = normalize_energy_frame(
                raw,
                timestamp_column=timestamp_column,
                demand_column=demand_column,
            )

        horizon = st.slider("Forecast horizon (hours)", 1, 168, 24)
        seasonal_period = st.select_slider(
            "Seasonal period (hours)",
            options=(12, 24, 48, 168),
            value=24,
            help="24 hours is the default daily energy-demand cycle.",
        )

        st.divider()
        st.markdown(f"[View source on GitHub]({REPOSITORY_URL})")
        st.caption("Forecast backend: deterministic seasonal-naive baseline")

    return frame, horizon, seasonal_period


def render_summary_metrics(summary: dict[str, float | int | str]) -> None:
    """Render concise operating metrics for the selected dataset."""

    first, second, third, fourth = st.columns(4)
    first.metric("Observations", f"{summary['observations']:,}")
    second.metric("Average demand", f"{summary['mean_demand']:,.1f} MW")
    third.metric("Peak demand", f"{summary['peak_demand']:,.1f} MW")
    fourth.metric("Minimum demand", f"{summary['minimum_demand']:,.1f} MW")


def render_architecture() -> None:
    """Render the public-demo and production-pipeline relationship."""

    st.graphviz_chart(
        """
digraph EnergyForecasting {
    rankdir=LR;
    node [shape=box, style=rounded];
    Data [label="CSV or Synthetic Data"];
    Validation [label="Schema & Quality Validation"];
    Baseline [label="Seasonal-Naive Baseline"];
    Streamlit [label="Streamlit Dashboard"];
    Training [label="LSTM / GRU / Transformer Training"];
    Registry [label="Model Artifacts / Registry"];
    API [label="FastAPI Serving"];
    CI [label="CI, Security, SBOM & Release Gates"];

    Data -> Validation;
    Validation -> Baseline;
    Baseline -> Streamlit;
    Validation -> Training;
    Training -> Registry;
    Registry -> API;
    Streamlit -> CI;
    API -> CI;
}
""",
        use_container_width=True,
    )


def main() -> None:
    """Render the Energy Demand Forecasting Streamlit application."""

    st.set_page_config(
        page_title="Energy Demand Forecasting DevMLOps",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        source_frame, horizon, seasonal_period = render_sidebar()
        frame = normalize_energy_frame(source_frame)
        settings = ForecastSettings(horizon=horizon, seasonal_period=seasonal_period)
        forecast = seasonal_naive_forecast(frame, settings)
        summary = summarize_dataset(frame)
    except (TypeError, ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"Unable to prepare the forecasting dataset: {exc}")
        st.stop()

    st.title("⚡ Energy Demand Forecasting — DevMLOps")
    st.subheader("Interactive, reproducible time-series forecasting and deployment-readiness demo")
    st.write(
        "Explore validated hourly demand data, deterministic multi-horizon forecasts, "
        "prediction intervals, and the operational controls used to ship the platform."
    )

    st.info(
        "This public dashboard uses a transparent seasonal-naive baseline so it can run "
        "without private datasets or model artifacts. The repository also contains the "
        "full LSTM, GRU, Transformer, training, API, container, and MLOps pathways."
    )

    render_summary_metrics(summary)

    overview_tab, forecast_tab, quality_tab, architecture_tab = st.tabs(
        ("Overview", "Forecast", "Data quality", "Architecture & operations")
    )

    with overview_tab:
        st.subheader("Recent hourly demand")
        recent = frame.tail(min(len(frame), 24 * 14)).set_index("timestamp")
        st.line_chart(recent[["demand"]], use_container_width=True)

        with st.expander("Dataset preview"):
            st.dataframe(frame.tail(24), use_container_width=True, hide_index=True)

    with forecast_tab:
        st.subheader(f"{horizon}-hour demand forecast")
        historical = frame.tail(min(len(frame), max(48, seasonal_period * 2)))[
            ["timestamp", "demand"]
        ].rename(columns={"demand": "actual"})
        future = forecast.rename(columns={"forecast": "predicted"})
        chart_frame = pd.concat(
            [
                historical.set_index("timestamp")[["actual"]],
                future.set_index("timestamp")[["predicted", "lower_bound", "upper_bound"]],
            ],
            axis=0,
        )
        st.line_chart(chart_frame, use_container_width=True)

        first, second, third = st.columns(3)
        first.metric("First forecast", f"{forecast['forecast'].iloc[0]:,.1f} MW")
        second.metric("Forecast peak", f"{forecast['forecast'].max():,.1f} MW")
        third.metric("Forecast average", f"{forecast['forecast'].mean():,.1f} MW")

        st.dataframe(forecast, use_container_width=True, hide_index=True)
        st.download_button(
            "Download forecast CSV",
            data=forecast.to_csv(index=False).encode("utf-8"),
            file_name="energy_demand_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with quality_tab:
        st.subheader("Validation evidence")
        quality_rows = pd.DataFrame(
            {
                "Check": (
                    "Required columns",
                    "Valid timestamps",
                    "Numeric demand",
                    "Non-negative demand",
                    "Unique timestamps after normalization",
                    "Minimum history",
                ),
                "Status": ("Passed", "Passed", "Passed", "Passed", "Passed", "Passed"),
            }
        )
        st.dataframe(quality_rows, use_container_width=True, hide_index=True)

        st.subheader("Dataset contract")
        st.json(summary)
        st.subheader("Forecast settings")
        st.json(asdict(settings))

    with architecture_tab:
        st.subheader("System architecture")
        render_architecture()

        st.subheader("Nine-tier deployment hygiene")
        st.markdown(
            """
1. **Source hygiene** — typed modules, syntax validation, Ruff, reproducible manifests.
2. **Test engineering** — unit, API, forecast-contract, and deployment smoke tests.
3. **Static quality** — CodeQL, compile validation, lint gates, coverage artifacts.
4. **Security engineering** — Gitleaks, Trivy, responsible disclosure policy.
5. **Supply-chain hygiene** — Dependabot, pip-audit, CycloneDX SBOM generation.
6. **Reproducible runtime** — multi-stage non-root container and slim deployment manifests.
7. **Continuous delivery** — Python matrix, API health, Streamlit health, container smoke tests.
8. **Release engineering** — semantic tags, GitHub Releases, GHCR image publishing.
9. **Operational governance** — health/metrics endpoints, changelog, contribution and rollback standards.
"""
        )

    st.divider()
    st.caption(
        "Portfolio demonstration by Corey Leath · Synthetic demo data is not a production forecast."
    )


if __name__ == "__main__":
    main()
