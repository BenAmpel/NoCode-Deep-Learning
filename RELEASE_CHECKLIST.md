# Release Checklist

## Source repository

- Confirm the project is inside a git repository
- Review `git status` and make sure only source files and release docs are staged
- Do not commit local-only directories such as `.venv/`, `dist/`, `build/`, `outputs/`, `.playwright-cli/`, or `packaging/vendor/`
- Make sure the top-level docs are current:
  - `README.md`
  - `SETUP.md`
  - packaging instructions
- Confirm version numbers are aligned across:
  - release notes
  - macOS package version
  - Windows installer version

## Functional validation

- Run:
  - `python3 install.py`
  - `python3 run_local.py --health-check-only`
- Launch the app locally and smoke test:
  - image tutorial
  - structured CSV preview
  - one short training run
  - one export
  - one inference test

## macOS dissemination

- Build unsigned package:
  - `python3 packaging/build_macos_installer.py --version <version>`
- If distributing broadly, sign and notarize:
  - `python3 packaging/sign_and_notarize_macos.py ...`
- Verify:
  - `pkgutil --check-signature dist/NoCode-DL-<version>-signed.pkg`

## Windows dissemination

- Build installer:
  - `python3 packaging/build_windows_installer.py --version <version>`
- Verify installer existence in `dist/Output/`
- If distributing broadly, sign the installer with your Windows code-signing workflow before release
- Test install on a clean Windows machine or VM

## GitHub release

- Push source only
- Attach platform installers as GitHub Release assets
- Include release notes covering:
  - supported platforms
  - known limitations
  - installation steps
  - trust/signing status

