#!/usr/bin/env bash
# Build a WilliamOS-owned Hermes image from the current Hermes checkout.
#
# Typical flow:
#   scripts/williamos-build-image.sh
#   scripts/williamos-build-image.sh --push
#
# Useful overrides:
#   IMAGE_REPO=ghcr.io/williamos-hq/hermes-agent TAG=my-tag scripts/williamos-build-image.sh
#   scripts/williamos-build-image.sh --platform linux/arm64 --tag local-arm64

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/williamos-build-image.sh [options]

Options:
  --push                 Push the image after a successful local build and smoke test.
  --no-smoke             Skip the container smoke tests.
  --image-repo REPO      Image repository. Default: ghcr.io/williamos-hq/hermes-agent
  --tag TAG              Image tag. Default: main-<origin-main-sha>-pr29302-<head-sha>
  --platform PLATFORM    Docker build platform. Default: linux/amd64
  -h, --help             Show this help.

Environment:
  IMAGE_REPO             Same as --image-repo.
  TAG                    Same as --tag.
  PLATFORM               Same as --platform.
  PR_NUMBER              PR number embedded in the default tag. Default: 29302.
  SMOKE_HOME             Host directory mounted at /opt/data for smoke tests.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_REPO="${IMAGE_REPO:-ghcr.io/williamos-hq/hermes-agent}"
PLATFORM="${PLATFORM:-linux/amd64}"
PR_NUMBER="${PR_NUMBER:-29302}"
TAG="${TAG:-}"
PUSH=0
SMOKE=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --push)
      PUSH=1
      ;;
    --no-smoke)
      SMOKE=0
      ;;
    --image-repo)
      [ "$#" -ge 2 ] || die "--image-repo requires a value"
      IMAGE_REPO="$2"
      shift
      ;;
    --image-repo=*)
      IMAGE_REPO="${1#*=}"
      ;;
    --tag)
      [ "$#" -ge 2 ] || die "--tag requires a value"
      TAG="$2"
      shift
      ;;
    --tag=*)
      TAG="${1#*=}"
      ;;
    --platform)
      [ "$#" -ge 2 ] || die "--platform requires a value"
      PLATFORM="$2"
      shift
      ;;
    --platform=*)
      PLATFORM="${1#*=}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
  shift
done

command -v docker >/dev/null 2>&1 || die "docker is required"
command -v git >/dev/null 2>&1 || die "git is required"

cd "$REPO_ROOT"

HEAD_SHA="$(git rev-parse --short HEAD)"
HEAD_FULL_SHA="$(git rev-parse HEAD)"
MAIN_SHA="$(git rev-parse --verify --short origin/main 2>/dev/null || git rev-parse --short HEAD)"
CREATED="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

if [ -z "$TAG" ]; then
  TAG="main-${MAIN_SHA}-pr${PR_NUMBER}-${HEAD_SHA}"
fi

IMAGE="${IMAGE_REPO}:${TAG}"

echo "Building Hermes image"
echo "  repo:     $IMAGE_REPO"
echo "  tag:      $TAG"
echo "  image:    $IMAGE"
echo "  platform: $PLATFORM"
echo "  revision: $HEAD_FULL_SHA"

docker build \
  --platform "$PLATFORM" \
  --label "org.opencontainers.image.created=$CREATED" \
  --label "org.opencontainers.image.revision=$HEAD_FULL_SHA" \
  --label "org.opencontainers.image.source=https://github.com/NousResearch/hermes-agent" \
  -t "$IMAGE" \
  .

if [ "$SMOKE" -eq 1 ]; then
  SMOKE_HOME="${SMOKE_HOME:-${TMPDIR:-/tmp}/williamos-hermes-smoke}"
  mkdir -p "$SMOKE_HOME"

  echo "Smoke testing $IMAGE"
  docker run --rm \
    -e "HERMES_UID=$(id -u)" \
    -e "HERMES_GID=$(id -g)" \
    -v "$SMOKE_HOME:/opt/data" \
    --entrypoint /opt/hermes/docker/entrypoint.sh \
    "$IMAGE" --help >/dev/null

  docker run --rm \
    -e "HERMES_UID=$(id -u)" \
    -e "HERMES_GID=$(id -g)" \
    -v "$SMOKE_HOME:/opt/data" \
    --entrypoint /opt/hermes/docker/entrypoint.sh \
    "$IMAGE" dashboard --help >/dev/null
fi

if [ "$PUSH" -eq 1 ]; then
  echo "Pushing $IMAGE"
  docker push "$IMAGE"
fi

echo "$IMAGE"
