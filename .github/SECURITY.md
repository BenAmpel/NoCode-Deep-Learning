# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅        |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a vulnerability, please email the maintainer directly or open a [GitHub Security Advisory](https://github.com/BenAmpel/NoCode-Deep-Learning/security/advisories/new).

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any suggested mitigations

You can expect an acknowledgement within 48 hours and a resolution timeline within 7 days for critical issues.

## Scope

This project runs **entirely locally** — no data is transmitted to external servers. The primary security surface is the installer bootstrapping process (downloading Python and dependencies on first launch). If you identify a supply-chain or integrity issue in this process, please report it.
