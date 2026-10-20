#!/usr/bin/env bash
# 指定したコミットに固定して git リポジトリを取得する。【新規】
# 切断後に入れ直したとき、別のコミット（観測の次元などが変わった版）になるのを防ぐ。
#   使い方: bash pin_clone.sh <URL> <フォルダ> <コミットのハッシュ>
set -euo pipefail
URL=$1; DIR=$2; REV=$3
if [ ! -d "$DIR/.git" ]; then
  git init -q "$DIR"
  git -C "$DIR" remote add origin "$URL"
  git -C "$DIR" fetch -q --depth 1 origin "$REV"
  git -C "$DIR" checkout -q FETCH_HEAD
fi
NOW=$(git -C "$DIR" rev-parse HEAD)
if [ "$NOW" != "$REV" ]; then
  echo "[warn] $DIR は $NOW です（指定 $REV と違う）。フォルダを消して入れ直してください。" >&2
fi
echo "$(basename "$DIR") @ ${NOW:0:7}"
