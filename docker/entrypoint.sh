#!/usr/bin/env sh
# First start: download data + models into the /workspace volume and build the frozen golden set.
# Later starts skip straight to the command. FAILFORGE_SKIP_SETUP=1 skips setup (CI, reference-only viewing).
set -e
if [ -z "$FAILFORGE_SKIP_SETUP" ] && [ ! -f "$FAILFORGE_HOME/data/manifests/golden.lock.json" ]; then
  echo "First start: downloading BDD100K and the models into $FAILFORGE_HOME (once) ..."
  failforge data download
  failforge data prepare
  failforge models download
fi
exec failforge "$@"
