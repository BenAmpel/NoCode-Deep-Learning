# Security Audit Notes

The `Security Scan` GitHub Actions workflow runs on every push and pull request
to `main`:

- **Dependency audit** with `pip-audit -r requirements.txt`. Any known
  vulnerability in a pinned package or anything it pulls in fails the job.
- **Static analysis** with `bandit`. High-severity findings fail the job;
  medium-severity findings are listed in the job summary and the uploaded report.

## Most recent local scan snapshot

Local scan performed on 2026-10-14 after the dependency upgrade:

- `pip-audit`: no known vulnerabilities (Python 3.12).
- `bandit`: no high-severity findings; 10 medium-severity findings.

The upgrade that cleared the dependency audit (previously 68 known
vulnerabilities across gradio, Pillow, torch, starlette, transformers,
python-multipart and onnx):

| Package | Before | After |
|---|---|---|
| gradio | 4.44.1 | 6.30.0 |
| transformers | 4.57.6 | 5.19.0 |
| torch / torchvision / torchaudio | 2.8.0 / 0.23.0 / 2.8.0 | 2.14.1 / 0.29.1 / 2.11.0 |
| Pillow | 10.4.0 | 12.3.0 |
| fastapi / starlette | 0.112.2 / 0.38.6 | 0.143.1 / 1.8.0 |
| onnx | 1.21.0 | 1.23.2 |
| python-multipart | 0.0.22 | 0.0.32 |

Gradio 5.x could not be used: its last release has advisories fixed only in
6.x and pins `Pillow<12`, which blocks the Pillow fixes.

Gradio usage analytics are disabled (`GRADIO_ANALYTICS_ENABLED=False` is set
before Gradio is imported), so the app makes no calls to `api.gradio.app`.

## Open medium-severity findings (`bandit`)

- Unpinned Hugging Face `from_pretrained()` downloads (no `revision=`) in the
  text, SHAP, export and inference paths.
- `torch.load(..., weights_only=False)` in `training/checkpointer.py`.
- `urllib` URL open without a scheme allow-list in `data_pipeline/network_utils.py`.

These should be treated as follow-up hardening work rather than ignored.
