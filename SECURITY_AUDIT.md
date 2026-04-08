# Security Audit Notes

This repository now includes a GitHub Actions workflow-based security scan:

- dependency audit via `pip-audit`
- static analysis via `bandit`

## Most recent local scan snapshot

Local scan performed on 2026-04-08 in the project environment found:

- 24 known dependency vulnerabilities across 4 packages
- 1 high-severity code-level finding before remediation in the macOS packaging checksum path
- multiple medium-severity findings related to unpinned model downloads and unsafe deserialization patterns

## Dependency findings surfaced by `pip-audit`

- `transformers==4.57.6`
- `gradio==4.44.1`
- `Pillow==10.4.0`
- `starlette==0.38.6`

## Important code-level findings surfaced by `bandit`

- unpinned `from_pretrained()` downloads in transformer and Whisper paths
- `torch.load(..., weights_only=False)` in checkpoint restore paths
- network/bootstrap download surfaces in packaging and tutorial setup

## Remediation status

- The macOS packager checksum verification was upgraded from MD5 to SHA-256.
- The repository now exposes a live `Security Scan` badge in the README.
- Remaining findings should be treated as follow-up hardening work rather than ignored.
