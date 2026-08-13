#!/usr/bin/env bash
#
# Recreates the throwaway workspace used by `examples/callback_approval.rs`
# (see `workspace_dir` / `setup_workspace` in that file), without needing to
# run the Rust example itself. Handy for poking at the fixture by hand or
# wiring it into a shell-based test.
#
# Workspace layout (mirrors `setup_workspace` in the example):
#   keep.txt    - must survive; the model is told not to touch it
#   tmp-a.log   - starts with `tmp-`; expected to be deleted
#   tmp-b.log   - starts with `tmp-`; expected to be deleted
#
# Usage:
#   script/setup-tmp-workspace.sh          # (re)create the workspace
#   script/setup-tmp-workspace.sh --clean  # remove it and exit
#   script/setup-tmp-workspace.sh --print  # print the workspace path only

set -euo pipefail

WORKSPACE_DIR="tmp"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--clean|--print|--help]

  (no args)   Wipe and recreate the demo workspace, then list its contents.
  --clean     Remove the demo workspace and exit.
  --print     Print the workspace path and exit (no filesystem changes).
  --help      Show this help.
EOF
}

case "${1:-}" in
  --help | -h)
    usage
    exit 0
    ;;
  --print)
    echo "$WORKSPACE_DIR"
    exit 0
    ;;
  --clean)
    rm -rf "$WORKSPACE_DIR"
    echo "removed: $WORKSPACE_DIR"
    exit 0
    ;;
  "")
    ;;
  *)
    echo "unknown option: $1" >&2
    usage >&2
    exit 1
    ;;
esac

rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"

printf 'important, do not delete\n' >"$WORKSPACE_DIR/keep.txt"
printf 'throwaway a\n' >"$WORKSPACE_DIR/tmp-a.log"
printf 'throwaway b\n' >"$WORKSPACE_DIR/tmp-b.log"

echo "workspace ready: $WORKSPACE_DIR"
ls -la "$WORKSPACE_DIR"
