## Summary

Describe the problem, the change, and the user or operational impact.

## Engineering design

Explain the selected design, alternatives considered, and important tradeoffs.

## Validation evidence

- [ ] Python 3.10 and 3.11 contract tests pass.
- [ ] Ruff correctness and syntax checks pass.
- [ ] Streamlit deployment smoke test passes when applicable.
- [ ] Container build and `/health` smoke test pass when applicable.
- [ ] Compose and Helm validation pass when applicable.
- [ ] New or changed behavior has automated tests.

Commands or evidence:

```text
Paste relevant commands, outputs, artifact links, or screenshots.
```

## Data and model impact

- Dataset schema changes:
- Feature changes:
- Leakage considerations:
- Model or artifact compatibility:
- Reproducibility impact:

## Security and supply-chain impact

Describe changes to inputs, secrets, dependencies, containers, permissions, network exposure, or release artifacts.

## Observability and operations

Describe metrics, logs, health checks, dashboards, alerts, capacity, or failure behavior.

## Deployment and rollback

Deployment steps:

Rollback or recovery steps:

## Documentation

- [ ] README or runbooks updated when behavior or deployment changes.
- [ ] CHANGELOG updated for user-visible or operational changes.
- [ ] No credentials, private datasets, local artifacts, or generated secrets are included.
