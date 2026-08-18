#!/usr/bin/env bash
# Back up Merge's irreplaceable state — everything under ~/.forge: her config,
# saved conversations, distilled memory-cards, and the honesty ledger. Writes a
# timestamped tarball and keeps the newest 30. Local redundancy: protects
# against a bad edit, corruption, or an accidental delete. For disk-failure
# safety, copy ~/backups/forge offsite (external drive / another machine).
set -euo pipefail

SRC="$HOME/.forge"
DEST="$HOME/backups/forge"
KEEP=30

[ -d "$SRC" ] || { echo "nothing to back up: $SRC missing" >&2; exit 1; }
mkdir -p "$DEST"

STAMP="$(date +%Y%m%d-%H%M%S)"
TARBALL="$DEST/forge-$STAMP.tar.gz"

# Skip the transient bits: the librarian's in-tray and any half-written temp files.
# Also grab her sim shelf (forge/sims) — those are hers, worth keeping even the
# ones she builds between git commits. `forge/sims` is added only if it exists.
EXTRA=""
[ -d "$HOME/forge/sims" ] && EXTRA="$EXTRA forge/sims"
[ -d "$HOME/forge/datasets" ] && EXTRA="$EXTRA forge/datasets"
tar -czf "$TARBALL" -C "$HOME" \
  --exclude='.forge/card-queue' \
  --exclude='.forge/*.tmp' \
  --exclude='.forge/.*.tmp' \
  --exclude='.forge/sessions/.*.tmp' \
  --exclude='forge/sims/*.tmp' \
  --exclude='forge/datasets/*.tmp' \
  .forge $EXTRA

# Retention: keep the newest $KEEP, drop the rest.
ls -1t "$DEST"/forge-*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f

echo "backed up → $TARBALL ($(du -h "$TARBALL" | cut -f1)); $(ls -1 "$DEST"/forge-*.tar.gz | wc -l) kept"

# Nightly shelf immune-system pass: re-validate every sim + dataset; a sim
# whose SELFTEST no longer passes gets demoted from ✓ in the catalog.
~/forge/.venv/bin/python -c "from forge import sims, datasets; print(sims.verify_shelf()); print(datasets.verify_shelf())" >> ~/forge/datasets/shelf-verify.log 2>&1
