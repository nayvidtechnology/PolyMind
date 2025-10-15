# Training Backends

Backends supported via orchestrator:
- local (default)
- azureml (stub)
- vertex (stub)
- sagemaker (stub)

Select backend in `configs/training/default.yaml`:
```yaml
trainer: local
```

Future: each backend module will submit jobs with configurable compute, data mounts, and checkpoint stores.
