Local MCP SDK patch backup for Atlassian OAuth

Why this exists
- Hermes repo changes can be tracked in git.
- The MCP SDK file below lives in site-packages and can be overwritten when dependencies are reinstalled.
- This directory keeps a tracked backup of the working patched file and a simple restore script.

Files
- oauth2.py.patched — known-good patched copy from this machine
- restore_oauth2_patch.sh — copies the patched file back into the active Hermes venv

When to use it
- After a Hermes update or dependency reinstall if Atlassian MCP OAuth starts failing again
- After recreating the venv

How to restore
- From the Hermes repo root, run:
  bash local_patches/mcp_sdk/restore_oauth2_patch.sh

What it restores
- venv/lib/python3.11/site-packages/mcp/client/auth/oauth2.py

Notes
- This is a recovery mechanism for the local MCP SDK patch until the upstream MCP SDK fix is fully released and adopted.
- The repo-tracked Hermes-side Atlassian fix in tools/mcp_oauth.py is already committed on branch local/tyler-hermes.
