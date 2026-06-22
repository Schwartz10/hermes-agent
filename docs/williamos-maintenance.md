# WilliamOS Hermes Maintenance

This repository tracks upstream `NousResearch/hermes-agent` while also carrying
WilliamOS-owned patches that may not be merged upstream immediately.

## Branch Model

- `main` mirrors `origin/main` from `NousResearch/hermes-agent`.
- `williamos/main` is the WilliamOS integration branch and the source for our
  Docker images.
- Upstream contribution branches stay as focused topic branches, for example
  `feat/session-tool-lifecycle-events`.
- WilliamOS-only changes should also live on focused topic branches before they
  are merged into `williamos/main`.

Keep `main` clean. Do not merge WilliamOS-only work into it.

Use merge commits on `williamos/main`. Do not routinely rebase or force-push the
integration branch because deployments and image tags should point at stable
history.

Use rebases on upstream PR branches. Those branches are review surfaces for
NousResearch, so keep them linear and force-push with lease after syncing.

## Current Patch Stack

As of this document, `williamos/main` carries:

- `feat/session-tool-lifecycle-events`: upstream PR
  `NousResearch/hermes-agent#38997`, streaming correlated tool lifecycle events
  from the session SSE endpoint.

When upstream merges one of our topic branches, sync `main` from `origin/main`,
merge `origin/main` into `williamos/main`, and remove any duplicate WilliamOS
patch if Git cannot resolve it naturally.

## Sync Flow

Update the upstream mirror:

```bash
git fetch origin
git switch main
git merge --ff-only origin/main
```

Update an upstream PR branch:

```bash
git switch feat/session-tool-lifecycle-events
git rebase origin/main
.venv/bin/python -m pytest tests/gateway/test_session_api.py -q -k session_chat_stream
git push --force-with-lease schwartz10 feat/session-tool-lifecycle-events
```

Update the WilliamOS integration branch:

```bash
git switch williamos/main
git merge origin/main
git merge --no-ff feat/session-tool-lifecycle-events
```

Resolve conflicts on `williamos/main`, run the relevant tests, then push:

```bash
git push schwartz10 williamos/main
```

## Docker Image Publishing

Use `scripts/williamos-build-image.sh` from `williamos/main`.

Default image repository:

```text
ghcr.io/williamos-hq/hermes-agent
```

Default platform:

```text
linux/amd64
```

Default tag shape:

```text
williamos-<origin-main-sha>-<head-sha>
```

Build and smoke-test locally:

```bash
scripts/williamos-build-image.sh
```

Build, smoke-test, and push an immutable tag:

```bash
scripts/williamos-build-image.sh --push
```

Build, smoke-test, push the immutable tag, and promote it to `latest`:

```bash
scripts/williamos-build-image.sh --push --latest
```

Use an explicit tag when publishing a release candidate or reproducing an older
build:

```bash
scripts/williamos-build-image.sh --push --tag williamos-2026-06-22
```

The smoke tests run `hermes --help` and `hermes dashboard --help` in the built
container with a temporary `/opt/data` mount. Set `SMOKE_HOME` to control that
host directory, or pass `--no-smoke` only when a separate validation already ran.
