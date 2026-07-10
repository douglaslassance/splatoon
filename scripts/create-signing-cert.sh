#!/usr/bin/env bash
# Create a stable, self-signed code-signing identity so macOS keeps privacy (TCC)
# grants — notably Photo Library access — across app rebuilds. Ad-hoc signatures
# are keyed by the binary's cdhash, which changes every build, so the system
# treats each rebuild as a new app and forgets the grant. A fixed identity fixes
# that. Run this ONCE; it modifies your login keychain and may ask for your
# password or a confirmation.
set -euo pipefail

CERT_NAME="Splatoon Dev"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

if security find-identity -v -p codesigning 2>/dev/null | grep -qF "$CERT_NAME"; then
  echo "Signing identity '$CERT_NAME' already exists. Nothing to do."
  exit 0
fi

echo "Creating self-signed code-signing identity '$CERT_NAME'..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/openssl.cnf" <<'CNF'
[ req ]
distinguished_name = dn
x509_extensions = v3
prompt = no
[ dn ]
CN = Splatoon Dev
[ v3 ]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
CNF

openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
  -keyout "$TMP/key.pem" -out "$TMP/cert.pem" -config "$TMP/openssl.cnf" >/dev/null 2>&1
openssl pkcs12 -export -inkey "$TMP/key.pem" -in "$TMP/cert.pem" \
  -name "$CERT_NAME" -out "$TMP/identity.p12" -passout pass: >/dev/null 2>&1

# Import the key + cert, granting codesign permission to use the private key.
security import "$TMP/identity.p12" -k "$KEYCHAIN" -P "" -T /usr/bin/codesign

# Trust the cert for code signing (a confirmation dialog may appear).
security add-trusted-cert -p codeSign -k "$KEYCHAIN" "$TMP/cert.pem" || true

echo
if security find-identity -v -p codesigning | grep -qF "$CERT_NAME"; then
  echo "Done. '$CERT_NAME' is ready."
  echo "Rebuild with scripts/build-app.sh, then grant Photo Library access once —"
  echo "it will now persist across rebuilds."
else
  echo "Identity created but not yet listed as valid for code signing."
  echo "Open Keychain Access, find '$CERT_NAME' under login, and set it to"
  echo "'Always Trust' for code signing, then rebuild."
fi
