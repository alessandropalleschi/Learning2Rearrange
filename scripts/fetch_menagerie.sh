#!/bin/bash
# The MuJoCo Menagerie UR5e (BSD-3, see its LICENSE) into assets/menagerie/universal_robots_ur5e, pinned to a commit.
set -euo pipefail
COMMIT=370c43965323c0944b433b202aac5058b5f907a9
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=$ROOT/assets/menagerie/universal_robots_ur5e
TMP=$(mktemp -d)
git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/m"
git -C "$TMP/m" sparse-checkout set universal_robots_ur5e
git -C "$TMP/m" checkout -q $COMMIT
mkdir -p "$DEST"
cp -r "$TMP/m/universal_robots_ur5e/ur5e.xml" "$TMP/m/universal_robots_ur5e/LICENSE" "$TMP/m/universal_robots_ur5e/README.md" \
      "$TMP/m/universal_robots_ur5e/assets" "$DEST/"
echo "google-deepmind/mujoco_menagerie@$COMMIT (universal_robots_ur5e), BSD-3 (LICENSE)" > "$DEST/SOURCE"
rm -rf "$TMP"
echo "fetched into $DEST"
