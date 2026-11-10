#!/bin/bash
# The MuJoCo Menagerie UR5e (BSD-3) and Franka FR3 (Apache-2.0) into assets/menagerie/<arm>, pinned to a commit.
set -euo pipefail
COMMIT=370c43965323c0944b433b202aac5058b5f907a9
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=$ROOT/assets/menagerie
TMP=$(mktemp -d)
git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/m"
git -C "$TMP/m" sparse-checkout set universal_robots_ur5e franka_fr3
git -C "$TMP/m" checkout -q $COMMIT
for ARM in universal_robots_ur5e franka_fr3; do
  mkdir -p "$DEST/$ARM"
  SRC="$TMP/m/$ARM"
  cp -r "$SRC"/*.xml "$SRC/LICENSE" "$SRC/README.md" "$SRC/assets" "$DEST/$ARM/"
  rm -f "$DEST/$ARM/scene.xml"
  echo "google-deepmind/mujoco_menagerie@$COMMIT ($ARM), see LICENSE" > "$DEST/$ARM/SOURCE"
done
rm -rf "$TMP"
echo "fetched into $DEST"
