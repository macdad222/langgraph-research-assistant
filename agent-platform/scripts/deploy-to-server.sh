#!/usr/bin/env bash
# Deploy the agent-platform source from this (Mac) git checkout to the
# runtime host (agent.lab.internal). The Mac is the source of truth; the
# server only runs the deployed code. Secrets and runtime data live ONLY on
# the server and are never overwritten (see excludes below).
#
# Usage:
#   scripts/deploy-to-server.sh                 # sync source only
#   scripts/deploy-to-server.sh --restart       # sync, then restart langgraph-app
#   scripts/deploy-to-server.sh --apply NAME    # sync, then apply model profile NAME and restart
#   scripts/deploy-to-server.sh --dry-run       # show what WOULD sync, change nothing
#
# Override defaults with env vars:
#   AGENT_DEPLOY_SERVER (default chris@agent.lab.internal)
#   AGENT_DEPLOY_DIR    (default /home/chris/dev/agent-platform)
set -euo pipefail

SERVER="${AGENT_DEPLOY_SERVER:-chris@agent.lab.internal}"
REMOTE_DIR="${AGENT_DEPLOY_DIR:-/home/chris/dev/agent-platform}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

DRY=""
ACTION=""
PROFILE=""
case "${1:-}" in
  --dry-run) DRY="--dry-run" ;;
  --restart) ACTION="restart" ;;
  --apply)   ACTION="apply"; PROFILE="${2:-}";
             [ -z "$PROFILE" ] && { echo "ERROR: --apply requires a profile name" >&2; exit 1; } ;;
  "")        ;;
  *)         echo "Unknown option: $1" >&2; exit 1 ;;
esac

# Never push secrets, runtime data, local backups, caches, or the stray nested dir.
EXCLUDES=(
  --exclude='.git'
  --exclude='.env'
  --exclude='.env.*'
  --exclude='*.bak-*'
  --exclude='credentials.txt'
  --exclude='data'
  --exclude='__pycache__'
  --exclude='*.pyc'
  --exclude='/agent-platform'
)

echo ">> Source: $SRC_DIR"
echo ">> Target: $SERVER:$REMOTE_DIR/"
[ -n "$DRY" ] && echo ">> DRY RUN (no changes)"

# Overlay sync (no --delete): safe with the live .env/data on the server.
rsync -az --human-readable --itemize-changes $DRY "${EXCLUDES[@]}" \
  "$SRC_DIR" "$SERVER:$REMOTE_DIR/"

if [ -n "$DRY" ]; then
  echo ">> Dry run complete. Nothing changed."
  exit 0
fi

case "$ACTION" in
  restart)
    echo ">> Restarting langgraph-app on $SERVER ..."
    ssh "$SERVER" "cd '$REMOTE_DIR' && bash scripts/agent-platform restart"
    ;;
  apply)
    echo ">> Applying profile '$PROFILE' and restarting on $SERVER ..."
    ssh "$SERVER" "cd '$REMOTE_DIR' && bash scripts/agent-platform model-profile apply '$PROFILE' --restart"
    ;;
  *)
    echo ">> Sync complete. Restart on the server when ready:"
    echo "     ssh $SERVER 'cd $REMOTE_DIR && bash scripts/agent-platform restart'"
    ;;
esac
