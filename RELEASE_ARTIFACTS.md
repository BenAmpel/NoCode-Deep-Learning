# Release Artifact Policy

## Commit to source control

- application source code
- packaging scripts
- fixtures required for documented demos
- workflow files
- documentation

## Do not commit to source control

- virtual environments
- local outputs and run history
- generated installers
- temporary build folders
- Playwright logs
- cached Python installers
- local signing artifacts or credentials

## Recommended distribution channels

- GitHub repository: source code only
- GitHub Releases: `.pkg`, `.exe`, release notes, checksums
- Optional external storage: very large offline package caches if you decide to keep them available

## Current dissemination guidance

- macOS:
  - distribute the signed/notarized `.pkg`
- Windows:
  - distribute the `.exe` only after you are satisfied with signing and install testing

