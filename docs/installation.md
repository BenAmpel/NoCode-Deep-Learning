# Installation

## Desktop Installers (recommended)

Download the latest installer from the [Releases page](https://github.com/BenAmpel/NoCode-Deep-Learning/releases/latest).

| Platform | File | Size | Notes |
|---|---|---|---|
| macOS (Intel + Apple Silicon) | `NoCode-DL-x.x.x.pkg` | ~800 KB | Signed & notarised; Gatekeeper-compatible |
| Windows (x64 + ARM64) | `NoCode-DL-Setup-x.x.x.exe` | ~2.3 MB | Detects architecture automatically |

### First launch

On first launch the app downloads **Python 3.12** and installs all runtime dependencies (~5 min on typical WiFi). Subsequent launches take 3–5 seconds. An internet connection is only required for this initial setup.

### macOS — Gatekeeper

If macOS warns that the package is from an "unidentified developer", right-click the `.pkg` and choose **Open**, then confirm. Alternatively, run:

```bash
xattr -d com.apple.quarantine NoCode-DL-x.x.x.pkg
```

### Windows — SmartScreen

Click **More info → Run anyway** if Windows SmartScreen prompts you. The installer is built and verified in native Windows CI on every release.

---

## Developer Install

### Requirements

- Python 3.12
- macOS 12+ or Windows 10+
- (Optional) NVIDIA GPU with CUDA, or Apple Silicon for MPS acceleration

### Steps

```bash
git clone https://github.com/BenAmpel/NoCode-Deep-Learning.git
cd NoCode-Deep-Learning
python3 install.py       # creates .venv and installs requirements.txt
python3 run_local.py     # launches at http://127.0.0.1:7860
```

The app opens in your default browser. Leave the terminal running — it hosts the local server.

---

## Building Installers

### macOS

```bash
python3 packaging/build_macos_installer.py --version 1.0.0
python3 packaging/sign_and_notarize_macos.py \
  --version 1.0.0 \
  --app-sign-identity "Developer ID Application: Your Name (TEAMID)" \
  --installer-sign-identity "Developer ID Installer: Your Name (TEAMID)" \
  --keychain-profile AC_PASSWORD
```

### Windows (cross-compiled on macOS via Wine)

```bash
brew install wine-stable
python3 packaging/build_windows_installer.py --version 1.0.0
```

See [architecture.md](architecture.md) for details on the two-stage bootstrap and cross-compilation approach.
