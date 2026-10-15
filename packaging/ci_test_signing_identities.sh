#!/bin/bash
# Create throwaway self-signed signing identities for CI, so the macOS
# installer job exercises the real signing path (codesign --timestamp with the
# hardened runtime, then productsign) instead of only an ad-hoc signature.
#
# CI ONLY: it trusts the certificates in the System keychain, which is fine on
# a disposable runner and not something to run on your own Mac.
#
# Usage: packaging/ci_test_signing_identities.sh <work-dir> >> "$GITHUB_ENV"
# Prints KEYCHAIN=, APP_IDENTITY= and INSTALLER_IDENTITY= lines.
set -euo pipefail

WORK="${1:?usage: ci_test_signing_identities.sh <work-dir>}"
mkdir -p "$WORK"
KEYCHAIN="$WORK/ci-signing.keychain-db"
PASSWORD="ci-throwaway"  # not a secret: the keychain lives only on this runner

security create-keychain -p "$PASSWORD" "$KEYCHAIN"
security set-keychain-settings "$KEYCHAIN"  # no auto-lock during the build
security unlock-keychain -p "$PASSWORD" "$KEYCHAIN"

make_identity() {
  local name="$1" eku="$2" marker="$3" base="$WORK/$4"
  cat > "$base.cnf" <<EOF
[req]
distinguished_name=dn
x509_extensions=ext
prompt=no
[dn]
CN=$name
[ext]
basicConstraints=critical,CA:false
keyUsage=critical,digitalSignature
extendedKeyUsage=critical,$eku
$marker=DER:0500
EOF
  openssl req -x509 -newkey rsa:2048 -nodes -days 2 -config "$base.cnf" \
    -keyout "$base.key" -out "$base.crt" 2>/dev/null
  openssl pkcs12 -export -inkey "$base.key" -in "$base.crt" -out "$base.p12" \
    -passout "pass:$PASSWORD" -legacy 2>/dev/null \
    || openssl pkcs12 -export -inkey "$base.key" -in "$base.crt" -out "$base.p12" -passout "pass:$PASSWORD"
  security import "$base.p12" -k "$KEYCHAIN" -P "$PASSWORD" \
    -T /usr/bin/codesign -T /usr/bin/productsign -T /usr/bin/productbuild >/dev/null
  sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "$base.crt"
}

# EKUs and Apple marker extensions as on real Developer ID certificates.
make_identity "NoCode-DL CI Application" codeSigning 1.2.840.113635.100.6.1.13 app
make_identity "NoCode-DL CI Installer" 1.2.840.113635.100.4.13 1.2.840.113635.100.6.1.14 installer
security set-key-partition-list -S apple-tool:,apple: -s -k "$PASSWORD" "$KEYCHAIN" >/dev/null

security find-identity -v "$KEYCHAIN" >&2
echo "KEYCHAIN=$KEYCHAIN"
echo "APP_IDENTITY=NoCode-DL CI Application"
echo "INSTALLER_IDENTITY=NoCode-DL CI Installer"
