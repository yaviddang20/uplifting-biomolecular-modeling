#!/usr/bin/env bash
# Build a kit's .sif with Apptainer alone (no Docker): bash tools/build_sif.sh <kit> [extra apptainer build args, e.g. --build-arg BUILD_JOBS=16]
# Run from anywhere; the build context is the repository root (it holds common/ and the kits).
# Needs: Apptainer >= 1.2 (build args), network access, and a build permission: root, --fakeroot (subuid/subgid set up by your admins),
# or Apptainer >= 1.3 unprivileged proot builds. The script tries --fakeroot unless APPTAINER_BUILD_FLAGS is set.
set -euo pipefail
kit="${1:?usage: build_sif.sh <kit> [apptainer build args...]}"; shift
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
[ -f "$kit/environment/Dockerfile" ] || { echo "no such kit: $kit" >&2; exit 2; }
python3 tools/docker2apptainer.py "$kit" >/dev/null     # regenerate so the def matches the Dockerfile
mkdir -p _jitcache                                     # optional pre-filled compile caches go here (see the kit's STOCK.md)
# large pulls/builds: keep Apptainer's cache and temp out of a small $HOME / /tmp if the site offers scratch
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-$HOME}/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH:-${TMPDIR:-/tmp}}/apptainer-tmp-$USER}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
out="${SIF_OUT:-$root/$kit.sif}"
flags="${APPTAINER_BUILD_FLAGS---fakeroot}"
[ -e "$out" ] && echo "note: $out exists and will be REPLACED (--force)"
# --no-mount tmp,home: the kits' Dockerfiles write scratch files under /tmp (e.g. /tmp/pip.txt). Under Docker that is a private /tmp; under
# Apptainer the HOST /tmp is mounted into %post, so on a shared login node such a path can already exist owned by another user and the build
# fails with "Permission denied". Keeping /tmp and $HOME out of the build also makes it independent of whatever is in them.
echo "building $out from $kit/environment/apptainer.native.def (flags: ${flags:-none})"
# shellcheck disable=SC2086
exec apptainer build $flags --force --no-mount tmp,home --warn-unused-build-args "$@" "$out" "$kit/environment/apptainer.native.def"
