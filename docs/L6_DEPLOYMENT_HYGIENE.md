# Energy Demand Forecasting — L6 Nine-Tier Deployment Hygiene

This document defines the repository's engineering maturity baseline and the automated evidence produced for each tier. The term **L6** is used here as a portfolio engineering standard, not as a claim of employment level or organizational certification.

## Tier 1 — Source Hygiene

**Objective:** keep source code readable, deterministic, reviewable, and resistant to configuration drift.

Controls:

- Python syntax compilation in CI.
- High-confidence Ruff correctness gates.
- Typed request, response, forecasting, and configuration contracts.
- Bounded API and dataset inputs.
- Canonical timestamp and demand schemas.
- Explicit runtime, development, training, and Streamlit dependency manifests.
- Reproducible synthetic data generation with fixed random seeds.

Evidence:

- `src/forecasting.py`
- `src/serve/app.py`
- `src/model.py`
- `requirements-api.txt`
- `requirements-dev.txt`
- `requirements.txt`
- `streamlit_demo/requirements.txt`

## Tier 2 — Test Engineering

**Objective:** verify expected behavior, edge cases, failure modes, and public contracts.

Controls:

- Python 3.10 and 3.11 compatibility matrix.
- API root, health, metrics, fallback, trained-model, validation, and error tests.
- Forecast reproducibility and output-contract tests.
- Missing-column, negative-demand, insufficient-history, and invalid-horizon tests.
- Coverage XML and JUnit evidence artifacts.
- Streamlit import and forecast-contract checks.

Evidence:

- `tests/test_api.py`
- `tests/test_forecasting.py`
- `.github/workflows/ci.yml`

## Tier 3 — Static Quality

**Objective:** identify correctness and security defects before runtime.

Controls:

- Ruff undefined-name and syntax-related checks.
- Python compile validation.
- CodeQL static analysis.
- Explicit Pydantic request and response models.
- Model-configuration validation.

Evidence:

- `.github/workflows/ci.yml`
- `.github/workflows/security.yml`
- `src/serve/app.py`
- `src/model.py`

## Tier 4 — Security Engineering

**Objective:** reduce the likelihood and impact of source, secret, dependency, and container vulnerabilities.

Controls:

- Gitleaks current-tree secret scanning.
- Trivy filesystem vulnerability reports.
- Trivy deployment-container vulnerability reports.
- Responsible vulnerability disclosure guidance.
- Non-root API runtime.
- Bounded request inputs and controlled service errors.

Evidence:

- `.github/workflows/security.yml`
- `SECURITY.md`
- `Dockerfile`

## Tier 5 — Supply-Chain Hygiene

**Objective:** make dependencies visible, reviewable, maintainable, and auditable.

Controls:

- Dependabot for Python, Streamlit, GitHub Actions, and Docker.
- `pip-audit` reports for deployment manifests.
- CycloneDX repository SBOM generation.
- Release-image provenance and SBOM attestations.
- Version-pinned dependency manifests.

Evidence:

- `.github/dependabot.yml`
- `.github/workflows/security.yml`
- `.github/workflows/release.yml`
- dependency manifests

## Tier 6 — Reproducible Runtime

**Objective:** make local, CI, container, and public-demo execution predictable.

Controls:

- Multi-stage API image.
- Minimal API dependency set.
- Non-root runtime identity.
- Explicit Uvicorn entry point.
- Container health check.
- Python 3.11 deployment pin.
- Lightweight Streamlit-specific dependency set.
- Artifact-independent public demo.
- Optional model artifacts mounted or supplied at runtime rather than baked into the image.

Evidence:

- `Dockerfile`
- `.python-version`
- `.streamlit/config.toml`
- `streamlit_demo/app.py`
- `streamlit_demo/requirements.txt`

## Tier 7 — Continuous Delivery

**Objective:** validate every integration path consistently before merge.

Controls:

- Pull-request and main-branch validation.
- Superseded-run cancellation.
- Multi-version quality and test matrix.
- Streamlit server health smoke test.
- Container build and live API health smoke test.
- Release-readiness contract that always executes and reports prerequisite states.

Evidence:

- `.github/workflows/ci.yml`

## Tier 8 — Release Engineering

**Objective:** create traceable, repeatable, and distributable release artifacts.

Controls:

- Semantic version tag trigger using `vMAJOR.MINOR.PATCH`.
- Generated GitHub Release notes.
- Source release archives that exclude local data and artifacts.
- GHCR container publishing.
- Container metadata, provenance, and SBOM generation.

Evidence:

- `.github/workflows/release.yml`
- `CHANGELOG.md`

## Tier 9 — Operational Governance

**Objective:** make ownership, operating assumptions, promotion criteria, and recovery expectations explicit.

Controls:

- Security disclosure process.
- Contribution and review standard.
- Semantic changelog.
- API liveness and Prometheus metrics endpoints.
- Explicit inference backend provenance.
- Public-demo synthetic-data disclosure.
- Streamlit deployment runbook.
- Auditable CI, security, dependency, coverage, test, and SBOM artifacts.

Evidence:

- `SECURITY.md`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `docs/STREAMLIT_DEPLOYMENT.md`
- `src/serve/app.py`
- `streamlit_app.py`

## Promotion Standard

A change is release-ready when:

1. Python 3.10 and 3.11 quality and contract tests pass.
2. Forecasting behavior remains deterministic for fixed inputs.
3. Streamlit starts and reports healthy in a deployment-like environment.
4. The API image builds and its live `/health` endpoint passes.
5. CodeQL and Gitleaks complete successfully.
6. Trivy, dependency-audit, coverage, test, and SBOM evidence is retained.
7. Release metadata and governance files are present.
8. Data, model, API, and deployment compatibility impacts are documented.
9. A practical rollback or recovery path exists for operationally significant changes.

Advisory findings should become focused remediation issues rather than being hidden, silently ignored, or conflated with unrelated feature work.
