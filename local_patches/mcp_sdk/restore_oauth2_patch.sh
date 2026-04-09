#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCHED="$REPO_ROOT/local_patches/mcp_sdk/oauth2.py.patched"
TARGET="$REPO_ROOT/venv/lib/python3.11/site-packages/mcp/client/auth/oauth2.py"
BACKUP_DIR="$REPO_ROOT/local_patches/mcp_sdk/backups"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

if [[ ! -f "$PATCHED" ]]; then
  echo "Patched source file not found: $PATCHED" >&2
  exit 1
fi

if [[ ! -f "$TARGET" ]]; then
  echo "Target MCP SDK file not found: $TARGET" >&2
  echo "If your venv path changed, update this script or recreate the expected venv first." >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
cp "$TARGET" "$BACKUP_DIR/oauth2.py.before-restore-$TIMESTAMP"
cp "$PATCHED" "$TARGET"

echo "Restored patched MCP SDK oauth2.py"
echo "Backup of previous target saved to: $BACKUP_DIR/oauth2.py.before-restore-$TIMESTAMP"
