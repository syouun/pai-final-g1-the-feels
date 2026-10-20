#!/usr/bin/env bash
# 学習ログ（チェックポイント・TensorBoard）をフォルダ間で同期する。【新規】
# 学習中はローカル（/content）に書き、定期的にドライブへ写す（ドライブへ直接書くと遅く、不安定なため）。
#   使い方: bash sync_logs.sh <元フォルダ> <先フォルダ>
set -euo pipefail
SRC=$1; DST=$2
[ -d "$SRC" ] || exit 0
mkdir -p "$DST"
if command -v rsync >/dev/null 2>&1; then
  rsync -a "$SRC"/ "$DST"/
else
  cp -ru "$SRC"/. "$DST"/
fi
