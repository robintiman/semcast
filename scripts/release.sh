#!/usr/bin/env bash
#
# Cut a release.
#
#   ./scripts/release.sh 0.2.1
#
# The script owns both the version bump and the tag, so the two can never
# disagree -- that mismatch is what broke the v0.2.0 release, where the tag
# said 0.2.0 but Cargo.toml still said 0.1.1 and dist had nothing to release.
#
# Pass --dry-run to run every check and see the plan without pushing anything.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RELEASE_BRANCH="main"

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }

DRY_RUN=0
VERSION=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    -*) die "unknown flag: $arg" ;;
    *) [ -n "$VERSION" ] && die "unexpected argument: $arg"; VERSION="$arg" ;;
  esac
done

[ -n "$VERSION" ] || die "usage: $0 <version> [--dry-run]   (e.g. $0 0.2.1)"

# Accept 1.2.3 and 1.2.3-rc.1, reject a leading "v" so the tag format stays ours.
case "$VERSION" in
  v*) die "pass the bare version without the leading 'v' (e.g. ${VERSION#v})" ;;
esac
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]] \
  || die "'$VERSION' is not a semver version"

TAG="v$VERSION"

# --- preflight ------------------------------------------------------------
# Everything below runs before a single byte is written or pushed.

step "Checking branch"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$CURRENT_BRANCH" = "$RELEASE_BRANCH" ] \
  || die "on '$CURRENT_BRANCH', releases must be cut from '$RELEASE_BRANCH'"

step "Checking working tree is clean"
git diff --quiet && git diff --cached --quiet \
  || die "working tree has uncommitted changes; commit or stash them first"

# A stale local main is how a release commit ends up missing work that was
# merged while you weren't looking. Refuse rather than silently release less
# than the user expects.
step "Fetching origin"
git fetch --quiet --tags --prune origin
LOCAL="$(git rev-parse "$RELEASE_BRANCH")"
REMOTE="$(git rev-parse "origin/$RELEASE_BRANCH")"
if [ "$LOCAL" != "$REMOTE" ]; then
  if git merge-base --is-ancestor "$LOCAL" "$REMOTE"; then
    die "local $RELEASE_BRANCH is behind origin/$RELEASE_BRANCH; run 'git pull --rebase' first"
  elif git merge-base --is-ancestor "$REMOTE" "$LOCAL"; then
    die "local $RELEASE_BRANCH has unpushed commits; push them and let CI pass first"
  else
    die "local and origin/$RELEASE_BRANCH have diverged; reconcile them first"
  fi
fi

step "Checking tag $TAG is free"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
  && die "tag $TAG already exists locally; delete it or pick a new version"
[ -z "$(git ls-remote --tags origin "refs/tags/$TAG")" ] \
  || die "tag $TAG already exists on origin; pick a new version"

CURRENT_VERSION="$(sed -n '/^\[package\]/,/^\[/{s/^version = "\(.*\)"/\1/p;}' Cargo.toml | head -1)"
[ -n "$CURRENT_VERSION" ] || die "could not read current version from Cargo.toml"
[ "$CURRENT_VERSION" != "$VERSION" ] \
  || die "Cargo.toml is already at $VERSION; pick a new version"
step "Bumping $CURRENT_VERSION -> $VERSION"

step "Checking CI is green on $(git rev-parse --short HEAD)"
if command -v gh >/dev/null 2>&1; then
  # An in-progress run reports conclusion as "", which jq's // does not treat
  # as empty -- read status and conclusion together so it can't be mistaken
  # for a failure.
  CI_STATE="$(gh run list --branch "$RELEASE_BRANCH" --commit "$(git rev-parse HEAD)" \
    --workflow CI --limit 1 --json status,conclusion \
    --jq '.[0] | if . == null then "none" else "\(.status):\(.conclusion)" end' 2>/dev/null || echo unknown)"
  case "$CI_STATE" in
    completed:success) ;;
    none|unknown) warn "no CI run found for this commit; continuing unverified" ;;
    completed:*) die "CI concluded '${CI_STATE#completed:}' on this commit; fix it before releasing" ;;
    *) warn "CI is still ${CI_STATE%%:*} on this commit; releasing before it finishes" ;;
  esac
else
  warn "gh not installed; skipping CI check"
fi

# --- apply ----------------------------------------------------------------

step "Writing version to Cargo.toml"
# Only the [package] version, never a dependency's.
sed -i.bak '/^\[package\]/,/^\[dependencies\]/ s/^version = ".*"/version = "'"$VERSION"'"/' Cargo.toml
rm -f Cargo.toml.bak

step "Updating Cargo.lock"
cargo update -p semcast --offline >/dev/null 2>&1 || cargo check --quiet

# Verify the bump actually landed where we think it did.
grep -q "^version = \"$VERSION\"$" Cargo.toml \
  || die "version bump did not apply cleanly to Cargo.toml; inspect it by hand"

# The exact check CI performs. If dist can't resolve the tag to a package
# version here, the release job would have failed the same way.
if command -v dist >/dev/null 2>&1; then
  step "Verifying dist can release $TAG"
  dist plan --tag="$TAG" >/dev/null \
    || die "dist cannot release $TAG -- run 'dist plan --tag=$TAG' to see why"
else
  warn "dist not installed; skipping the pre-flight dist plan"
  warn "install it with: cargo install cargo-dist --version 0.28.7"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  step "Dry run -- reverting local changes"
  git checkout -- Cargo.toml Cargo.lock
  printf '\nAll checks passed. Re-run without --dry-run to release %s.\n' "$TAG"
  exit 0
fi

printf '\nAbout to release \033[1m%s\033[0m from %s (%s).\n' \
  "$TAG" "$RELEASE_BRANCH" "$(git rev-parse --short HEAD)"
read -r -p "Push the release commit and tag? [y/N] " reply
[[ "$reply" =~ ^[Yy]$ ]] || { git checkout -- Cargo.toml Cargo.lock; die "aborted"; }

step "Committing"
git add Cargo.toml Cargo.lock
git commit -m "Release $TAG"

step "Pushing $RELEASE_BRANCH"
# --atomic so a rejected branch push doesn't leave the tag pointing at a commit
# origin never accepted.
git tag -a "$TAG" -m "Release $TAG"
if ! git push --atomic origin "$RELEASE_BRANCH" "refs/tags/$TAG"; then
  git tag -d "$TAG" >/dev/null
  git reset --hard HEAD~1
  die "push rejected (origin likely moved); local state rolled back -- pull and retry"
fi

printf '\n\033[32mReleased %s\033[0m\n' "$TAG"
if command -v gh >/dev/null 2>&1; then
  printf 'Watch it: gh run watch --repo %s "$(gh run list --workflow Release --limit 1 --json databaseId --jq \x27.[0].databaseId\x27)"\n' \
    "$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || echo robintiman/semcast)"
fi
