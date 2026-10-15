#!/bin/bash
# BUILD a NoCode-DL macOS release into releases/staging/<version>/ and nothing else.
#
#   NCDL_VERSION=0.1.0 scripts/build-release.sh
#
# Same release path as the other Purplelink apps: the .app is signed with the Developer ID identity and the hardened
# runtime, notarized with the keychain profile "moderntex", stapled, and packed into a DMG that is signed, notarized and
# stapled too. A DMG needs only the Developer ID Application certificate (a .pkg would also need Developer ID Installer).
# The .app installs Python 3.12 from python.org on first launch when it's missing, which the .pkg did at install time.
# The result waits in staging; this script publishes nothing.
#
#   NCDL_SIGN_IDENTITY   default: Developer ID Application: Purplelink LLC (864CY4267E)
#   NCDL_NOTARY_PROFILE  default: moderntex
#   NCDL_RELEASE_DD      work folder, on the INTERNAL disk
set -euo pipefail
cd "$(dirname "$0")/.."

die() { echo "ERROR: $*" >&2; exit 1; }
plist_value() { /usr/libexec/PlistBuddy -c "Print :$2" "$1" 2>/dev/null || true; }

VERSION="${NCDL_VERSION:?set NCDL_VERSION=x.y.z}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "NCDL_VERSION must be x.y.z"
SIGN_IDENTITY="${NCDL_SIGN_IDENTITY:-Developer ID Application: Purplelink LLC (864CY4267E)}"
NOTARY_PROFILE="${NCDL_NOTARY_PROFILE:-moderntex}"
RELEASE_DD="${NCDL_RELEASE_DD:-/private/tmp/nocode-dl-release-dd}"
STAGING="releases/staging"
APP_NAME="NoCode-DL"
DMG_FILE="$APP_NAME-$VERSION.dmg"

# --- Preconditions ---------------------------------------------------------------------------------------------
[ -z "$(git status --porcelain --untracked-files=no)" ] || die "uncommitted changes; a release must come from a commit."
xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1 \
    || die "no working notary profile '$NOTARY_PROFILE'; a public build must be notarized (xcrun notarytool store-credentials $NOTARY_PROFILE ...)"
case "$(security find-identity -v -p codesigning)" in *"$SIGN_IDENTITY"*) ;; *) die "signing identity not in the keychain: $SIGN_IDENTITY" ;; esac

DEST="$STAGING/$VERSION"
WORK="$RELEASE_DD/work"
APP="$WORK/$APP_NAME.app"
NOTARY_IDS=""
rm -rf "${WORK:?}"; mkdir -p "$WORK"

# --- Build and sign the app: Developer ID, hardened runtime, secure timestamp -------------------------------------
echo "=== build $APP_NAME $VERSION ==="
python3 packaging/build_macos_installer.py --version "$VERSION" --app-sign-identity "$SIGN_IDENTITY" --app-only "$WORK"
[ -d "$APP" ] || die "the build produced no $APP"
[ "$(plist_value "$APP/Contents/Info.plist" CFBundleShortVersionString)" = "$VERSION" ] || die "the app's version is not $VERSION"
codesign --verify --deep --strict --verbose=2 "$APP"
SIGINFO="$(codesign -dvv "$APP" 2>&1)"
case "$SIGINFO" in *"flags=0x10000(runtime)"*) ;; *) die "the app is not signed with the hardened runtime" ;; esac
case "$SIGINFO" in *"Timestamp="*) ;; *) die "the app's signature has no secure timestamp" ;; esac
case "$SIGINFO" in *"Authority=$SIGN_IDENTITY"*) ;; *) die "the app is not signed by $SIGN_IDENTITY" ;; esac

# --- Notarize --------------------------------------------------------------------------------------------------
notarize() {   # file label
    local out id status
    out="$RELEASE_DD/notary-$2.log"
    xcrun notarytool submit "$1" --keychain-profile "$NOTARY_PROFILE" --wait > "$out" 2>&1 || true
    id="$(grep -m1 -E '^ *id:' "$out" | awk '{print $2}')"
    status="$(grep -E '^ *status:' "$out" | tail -1 | awk '{print $2}')"
    echo "notarization ($2): submission $id, status ${status:-unknown}"
    NOTARY_IDS="$NOTARY_IDS$2=$id:$status "
    if [ "$status" != "Accepted" ]; then
        cat "$out" >&2
        [ -z "$id" ] || xcrun notarytool log "$id" --keychain-profile "$NOTARY_PROFILE" >&2 || true
        die "notarization of $2 was not accepted"
    fi
}
ZIP="$WORK/$APP_NAME.zip"
ditto -c -k --keepParent "$APP" "$ZIP"
notarize "$ZIP" app
rm -f "$ZIP"
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"
spctl --assess --type execute -vv "$APP"

# --- DMG -------------------------------------------------------------------------------------------------------
DMG="$WORK/$DMG_FILE"
STAGE_DIR="$WORK/dmg-root"
mkdir -p "$STAGE_DIR"
ditto "$APP" "$STAGE_DIR/$APP_NAME.app"
ln -s /Applications "$STAGE_DIR/Applications"
# "Resource busy" appears now and then right after stapling (the system is still scanning the app); retry.
for attempt in 1 2 3 4 5 6; do
    hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE_DIR" -ov -format UDZO "$DMG" >/dev/null 2>&1 && break
    [ "$attempt" -lt 6 ] || die "hdiutil create failed six times"
    echo "hdiutil create failed (attempt $attempt); retrying" >&2; sleep 10
done
rm -rf "${STAGE_DIR:?}"
for attempt in 1 2 3 4 5 6; do
    codesign --force --sign "$SIGN_IDENTITY" --timestamp "$DMG" && break
    [ "$attempt" -lt 6 ] || die "codesign of the DMG failed six times"
    echo "codesign failed (attempt $attempt); retrying" >&2; sleep 10
done
notarize "$DMG" dmg
xcrun stapler staple "$DMG"

# --- Verify what will be handed out ----------------------------------------------------------------------------
echo "=== verify $DMG_FILE ==="
xcrun stapler validate "$DMG"
spctl --assess --type open --context context:primary-signature -vv "$DMG"
MOUNT="$WORK/mount"
mkdir -p "$MOUNT"
hdiutil attach "$DMG" -readonly -nobrowse -noautoopen -mountpoint "$MOUNT" >/dev/null
detach() { hdiutil detach "$MOUNT" -quiet 2>/dev/null || hdiutil detach "$MOUNT" -force -quiet 2>/dev/null || true; }
trap detach EXIT
[ "$(plist_value "$MOUNT/$APP_NAME.app/Contents/Info.plist" CFBundleShortVersionString)" = "$VERSION" ] \
    || die "the DMG's app has the wrong version"
[ -L "$MOUNT/Applications" ] || die "the DMG has no Applications link"
codesign --verify --deep --strict "$MOUNT/$APP_NAME.app"
spctl --assess --type execute -vv "$MOUNT/$APP_NAME.app"
xcrun stapler validate "$MOUNT/$APP_NAME.app"
detach; trap - EXIT
rmdir "$MOUNT" 2>/dev/null || true

# --- Stage -----------------------------------------------------------------------------------------------------
mkdir -p "$DEST"
cp -f "$DMG" "$DEST/$DMG_FILE"
{
    echo "version=$VERSION"
    echo "commit=$(git rev-parse --short HEAD)"
    echo "notarized=1"
    echo "notarization=$NOTARY_IDS"
    echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "size=$(stat -f%z "$DEST/$DMG_FILE")"
    echo "sha256=$(shasum -a 256 "$DEST/$DMG_FILE" | cut -d' ' -f1)"
} > "$DEST/macos.manifest"
echo "Staged: $DEST/$DMG_FILE"
cat "$DEST/macos.manifest"
