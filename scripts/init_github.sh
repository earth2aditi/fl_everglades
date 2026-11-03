#!/bin/bash
# Create a GitHub repo for the Florida Deepbiosphere project and push it.
# Run on the BlueHive LOGIN node (needs internet), from anywhere:
#   bash /scratch/ajasapar/deepbiosphere/scripts/init_github.sh [repo-name] [private|public]
# Safe to rerun: it reuses the existing git repo / remote.
set -eo pipefail

ROOT=/scratch/ajasapar/deepbiosphere
REPO_NAME=${1:-deepbiosphere-florida}
VIS=${2:-private}
cd "$ROOT"

# ---- 0. preflight ----------------------------------------------------------
[ -f README.md ] && [ -f .gitignore ] || {
  echo "Copy README.md and .gitignore (from the chat) into $ROOT first."; exit 1; }
for f in scripts/figures_fl.sh scripts/figures_fl.py; do
  [ -f "$f" ] || echo "WARNING: $f is missing (copy it from the chat attachments; rerun this script afterwards)."
done

# ---- 1. GitHub CLI (standalone binary in tools/, leaves the conda env alone) -
GH=$(command -v gh || true)
if [ -z "$GH" ]; then
  GH=$(ls -d "$ROOT"/tools/gh_*_linux_amd64/bin/gh 2>/dev/null | tail -1 || true)
fi
if [ -z "$GH" ]; then
  echo "Installing gh into $ROOT/tools ..."
  mkdir -p tools
  TAG=$(curl -fsSL https://api.github.com/repos/cli/cli/releases/latest \
        | python3 -c 'import sys,json; print(json.load(sys.stdin)["tag_name"])')
  V=${TAG#v}
  curl -fsSL "https://github.com/cli/cli/releases/download/${TAG}/gh_${V}_linux_amd64.tar.gz" | tar -xz -C tools
  GH="$ROOT/tools/gh_${V}_linux_amd64/bin/gh"
fi
echo "gh: $GH"

if ! "$GH" auth status >/dev/null 2>&1; then
  echo "Log in to GitHub: open the URL it prints in your own browser and enter the code."
  "$GH" auth login --hostname github.com --git-protocol https --web
fi
"$GH" auth setup-git
LOGIN=$("$GH" api user --jq .login)

# ---- 2. Florida patch against upstream Deepbiosphere -----------------------
mkdir -p patches
: > patches/florida.patch
while IFS= read -r orig; do
  f=${orig%.ca_orig}; rel=${f#Deepbiosphere/}
  diff -u --label "a/$rel" --label "b/$rel" "$orig" "$f" >> patches/florida.patch || [ $? -eq 1 ]
done < <(find Deepbiosphere -name '*.ca_orig' | sort)
echo "patches/florida.patch: $(grep -c '^+++ ' patches/florida.patch) file(s)"

if [ -d Deepbiosphere/.git ]; then
  {
    echo "remote: $(git -C Deepbiosphere remote get-url origin 2>/dev/null || echo unknown)"
    echo "commit: $(git -C Deepbiosphere rev-parse HEAD)"
    echo "apply:  patch -p1 -d Deepbiosphere < patches/florida.patch"
  } > patches/UPSTREAM.txt
  # any edits to tracked upstream files beyond the .ca_orig set
  git -C Deepbiosphere diff > patches/deepbiosphere_worktree.diff
  [ -s patches/deepbiosphere_worktree.diff ] || rm -f patches/deepbiosphere_worktree.diff
else
  echo "NOTE: Deepbiosphere/ is not a git clone; fill in patches/UPSTREAM.txt by hand."
  [ -f patches/UPSTREAM.txt ] || echo "remote: (fill in)  commit: (fill in)" > patches/UPSTREAM.txt
fi

# ---- 3. git repo + identity ------------------------------------------------
[ -d .git ] || git init -q -b main
if ! git config user.email >/dev/null; then
  ID=$("$GH" api user --jq .id)
  git config user.name  "$LOGIN"
  git config user.email "${ID}+${LOGIN}@users.noreply.github.com"
  echo "Set repo-local git identity to $LOGIN (GitHub noreply email)."
fi

git add -A

# ---- 4. safety checks before anything leaves the cluster -------------------
BIG=$(git diff --cached --name-only -z | xargs -0 -r du -k | awk '$1>50000')
if [ -n "$BIG" ]; then
  echo "Staged files over 50 MB (add them to .gitignore, then rerun):"; echo "$BIG"
  git reset -q; exit 1
fi

SECRETS=$(git grep --cached -nIiE '(password|passwd|pwd=|token|secret|api[_-]?key)' -- . ':!README.md' ':!patches/' :!scripts/init_github.sh || true)
if [ -n "$SECRETS" ]; then
  echo "Possible credentials in staged files (e.g. a GBIF password in download_fl_data.sh):"
  echo "$SECRETS"
  echo "Move them to an env var or a git-ignored file, then rerun."
  git reset -q; exit 1
fi

echo; echo "About to commit $(git diff --cached --name-only | wc -l) files:"
git diff --cached --stat | tail -40
read -rp "Commit and push to github.com/$LOGIN/$REPO_NAME ($VIS)? [y/N] " ok
[ "$ok" = y ] || [ "$ok" = Y ] || { git reset -q; echo "Aborted; nothing committed."; exit 1; }

if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -q -F - <<'EOF'
Florida Deepbiosphere pipeline: scripts, patches, small outputs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SduegoEFEpdMDyMwDEEnUJ
EOF
fi

# ---- 5. create repo / push --------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  "$GH" repo create "$REPO_NAME" "--$VIS" --source . --remote origin --push \
    --description "Deepbiosphere species distribution models adapted to Florida (NAIP 2021 + GBIF/iNaturalist)"
fi
echo "Done: https://github.com/$LOGIN/$REPO_NAME"
