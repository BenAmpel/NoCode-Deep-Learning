# Setup

```bash
python3 install.py
```

Then launch:
```bash
python3 run_local.py
```

Opens at http://127.0.0.1:7860

## macOS Installer

Build a `.pkg` installer:

```bash
python3 packaging/build_macos_installer.py --version 1.0.0
```

This creates a branded macOS installer in `dist/` that installs `NoCode-DL.app` into `/Applications` and embeds the official Python 3.12 installer so Python is installed automatically if it is missing.

By default this build step creates an unsigned package. Sign and notarize separately for public distribution.

### Signing and notarization

Once your Apple Developer credentials are configured locally, build a signed and notarized installer with:

```bash
python3 packaging/sign_and_notarize_macos.py \
  --version 1.0.0 \
  --app-sign-identity "Developer ID Application: Your Name (TEAMID)" \
  --installer-sign-identity "Developer ID Installer: Your Name (TEAMID)" \
  --keychain-profile AC_PASSWORD
```

This signs the app bundle, signs the installer package, submits it with `notarytool`, and staples the notarization ticket.
