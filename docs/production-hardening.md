# Production hardening checklist

This project is an educational ABP estimation service and is not a certified medical device. The checklist below records controls required before any deployment beyond a local demonstration.

## API boundary

Validate JSON content type, finite numeric values, exact signal length, maximum request size, and model output shape. Return stable error codes without exposing exception text. Add authentication, per-client rate limiting, request IDs, access logging with sensitive payloads excluded, and dependency vulnerability scanning.

## Model operations

Record model artifact checksums, training-data version, feature order, scaler versions, and evaluation metrics. Load artifacts through explicit configuration and fail health checks when the bundle is incomplete. Add model-version labels and a controlled rollback path.

## Safety and observability

Keep the non-clinical disclaimer visible, do not accept clinical decisions as a supported use case, and add monitoring for latency, error rate, input drift, and output distribution. Any clinical use requires an independent safety, privacy, and validation review.

## Release gate

A release must pass `pytest`, `ruff`, dependency scanning, container smoke tests, and contract tests against a staging service. Deployment credentials must be supplied by the platform secret store, never committed to Git.
