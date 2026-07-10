# Security Policy

## Supported versions

Security fixes are applied to the latest tagged release and the `main` branch.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Use GitHub private vulnerability reporting when it is enabled, or contact the repository owner privately with:

- the affected component and version;
- clear reproduction steps;
- the expected security impact;
- any known mitigations or suggested remediation.

Please avoid including production credentials, private datasets, or unnecessary personal information in a report.

## Response process

Confirmed reports are triaged by severity, reproduced in an isolated environment, remediated on a focused branch, and documented after a fix is available. Security-sensitive details may remain private until affected users have a reasonable opportunity to update.

## Automated controls

Energy Demand Forecasting DevMLOps uses:

- CodeQL static analysis;
- Gitleaks secret scanning;
- Trivy filesystem and container reporting;
- `pip-audit` dependency reports;
- Dependabot updates;
- CycloneDX SBOM generation;
- multi-version tests;
- Streamlit and container smoke tests;
- release-readiness validation.

Automated tools supplement, rather than replace, engineering review and threat modeling.
