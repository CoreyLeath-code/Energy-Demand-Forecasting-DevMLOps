# Kubernetes Deployment Notes

The manifests in this directory provide a secure baseline for the FastAPI inference runtime.

## Apply the reference manifests

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

Configure `k8s/ingress.yaml` with a real hostname before applying it.

## Model artifacts

The API remains healthy when no trained artifact is mounted and clearly labels predictions from its deterministic fallback backend.

For trained-model inference, provide a compatible artifact at:

```text
/app/models/energy_forecast_model.pkl
```

Use an environment-appropriate read-only volume, object-storage synchronization process, model registry integration, or init container. Do not bake private model artifacts into public images.

## Secrets

No plaintext or base64-encoded secret manifest is committed to this repository.

Use one of the following deployment-specific approaches:

- External Secrets Operator;
- Sealed Secrets;
- cloud-provider secret stores;
- a GitOps platform with encrypted secret support;
- a namespace-scoped secret created through a secure deployment pipeline.

Do not commit credentials, tokens, kubeconfigs, registry passwords, Snowflake credentials, Kafka credentials, or MLflow authentication material.

## Runtime security baseline

The deployment uses:

- a non-root UID and GID;
- `RuntimeDefault` seccomp;
- `allowPrivilegeEscalation: false`;
- a read-only root filesystem;
- all Linux capabilities dropped;
- disabled automatic service-account token mounting;
- bounded CPU and memory resources;
- startup, readiness, and liveness probes;
- Prometheus scrape annotations.

These settings are a baseline and should be complemented by namespace policies, network policies, admission controls, image verification, workload identity, and environment-specific observability.
