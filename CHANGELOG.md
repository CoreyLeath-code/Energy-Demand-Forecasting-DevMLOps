# Changelog

All notable changes to Energy Demand Forecasting DevMLOps are documented here.

The project follows Semantic Versioning and the Keep a Changelog format.

## [Unreleased]

### Added

- Artifact-independent Streamlit Community Cloud dashboard.
- Reproducible synthetic hourly demand generator.
- CSV upload, schema validation, data-quality reporting, and forecast downloads.
- Deterministic seasonal-naive forecasting with bounded trend adjustment and uncertainty intervals.
- Dataset summary and frequency-inference utilities.
- Root, health, metrics, and validated prediction API contracts.
- Transparent trained-model versus deterministic-baseline inference provenance.
- Python 3.10 and 3.11 CI matrix.
- API, forecasting, validation, and failure-mode tests.
- Coverage XML and JUnit evidence artifacts.
- Streamlit deployment and container health smoke tests.
- CodeQL, Gitleaks, Trivy filesystem and container reports, pip-audit, Dependabot, and CycloneDX SBOM automation.
- Semantic tag-driven GitHub Releases and GHCR image publishing.
- Security, contribution, Streamlit deployment, and nine-tier deployment-hygiene documentation.
- Slim API, development, full-training, and Streamlit dependency manifests.

### Changed

- Hardened the FastAPI service with bounded inputs, lazy artifact loading, explicit service metadata, Prometheus instrumentation, and controlled error responses.
- Reworked the API container as a multi-stage, non-root runtime with an explicit health check.
- Normalized model configuration so uppercase model names and the existing `hidden_units` YAML key are supported.
- Added model-architecture validation and safe single-layer recurrent dropout behavior.
- Rebuilt the Streamlit application so public deployment does not require private data, model checkpoints, scalers, or PyTorch.
- Updated repository links and contribution instructions to the current `CoreyLeath-code` owner.

## [0.1.0] - 2025-06-01

### Added

- Initial DevMLOps forecasting pipeline.
- Data preprocessing and feature engineering.
- XGBoost and deep-learning model scaffolding.
- FastAPI serving.
- Kubernetes, Helm, DVC, MLflow, Prometheus, Grafana, Terraform, and Ansible infrastructure foundations.
- Initial GitHub Actions validation.

[Unreleased]: https://github.com/CoreyLeath-code/Energy-Demand-Forecasting-DevMLOps/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CoreyLeath-code/Energy-Demand-Forecasting-DevMLOps/releases/tag/v0.1.0
