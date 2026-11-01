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
# Default the caches beside the repo (KIT_BASE: the directory holding the checkout, weights/, kit_out/ -- the same
# base tools/boltz_compat/boltz resolves), not $HOME. A build unpacks several GB, and a site that does not set
# SCRATCH leaves $HOME as the fallback, where a home quota of a few GB fails the build partway through.
base="$(dirname "$root")"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-$base}/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH:-$base}/apptainer-tmp-$USER}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
out="${SIF_OUT:-$root/$kit.sif}"
# --fakeroot needs subuid/subgid ranges your admins set up; without them apptainer's own unprivileged path
# builds fine, so do not ask for a flag the host cannot honour. Set APPTAINER_BUILD_FLAGS to override either way.
if [ -n "${APPTAINER_BUILD_FLAGS+x}" ]; then flags="$APPTAINER_BUILD_FLAGS"
elif grep -q "^$USER:" /etc/subuid 2>/dev/null; then flags="--fakeroot"
else flags=""; echo "note: $USER is not in /etc/subuid, building without --fakeroot"; fi
[ -e "$out" ] && echo "note: $out exists and will be REPLACED (--force)"
# A PRIVATE /tmp for the build: the kits' Dockerfiles write scratch files under /tmp (e.g. /tmp/pip.txt). Under Docker that is a private
# /tmp; under Apptainer the HOST /tmp is mounted into %post, so on a shared login node such a path can already exist owned by another user
# and the build dies with "Permission denied". `apptainer build` has no --no-mount, so bind an empty directory over /tmp instead.
btmp="$APPTAINER_TMPDIR/buildtmp-$kit.$$"; mkdir -p "$btmp"
trap 'rm -rf "$btmp"' EXIT
echo "building $out from $kit/environment/apptainer.native.def (flags: ${flags:-none})"
# shellcheck disable=SC2086
apptainer build $flags --force --bind "$btmp":/tmp --warn-unused-build-args "$@" "$out" "$kit/environment/apptainer.native.def"
rc=$?
exit $rc
