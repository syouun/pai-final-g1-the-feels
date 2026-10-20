#!/usr/bin/env bash
# 3つの動画を横に並べた比較動画を作る（高さ720pxにそろえ、30fpsで出力）。【改訂版】
#   左  : GVHMR が推定した人の動き（メッシュ表示。元動画の人物は映らない）
#   中央: GMR でリターゲットしたお手本（物理なし）
#   右  : 強化学習した方策で踊る G1（物理シミュレーション）
# 第4引数に元動画を渡すと、その音声を付ける（音声なしにしたい場合は省略）。
#
# 区間を切り出したお手本（例: 全編の 59.815〜80.78 秒）と並べるときは、環境変数 START に
# 区間の開始秒を入れる。左（全編の人のメッシュ動画）と音声だけが START 秒から始まるようにずらす。
# （中央と右は、すでに区間の先頭から始まっている）
#
# 使い方:
#   START=59.815 bash make_comparison.sh human_full.mp4 retarget_20s.mp4 policy_20s.mp4 [audio_full.mp4] out.mp4
set -euo pipefail

if [ $# -eq 4 ]; then
  HUMAN=$1; RETARGET=$2; POLICY=$3; AUDIO=""; OUT=$4
elif [ $# -eq 5 ]; then
  HUMAN=$1; RETARGET=$2; POLICY=$3; AUDIO=$4; OUT=$5
else
  echo "usage: [START=秒] $0 human.mp4 retarget.mp4 policy.mp4 [audio_source.mp4] out.mp4" >&2
  exit 1
fi
START=${START:-0}

dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
echo "[info] 長さ: お手本 $(dur "$RETARGET") 秒 / 物理 $(dur "$POLICY") 秒"
python3 - "$(dur "$RETARGET")" "$(dur "$POLICY")" <<'EOF'
import sys
r, p = map(float, sys.argv[1:3])
if p < r - 0.5:
    print(f"[warn] 物理シミュレーションの動画がお手本より {r - p:.1f} 秒短い（途中で転倒した方策）。"
          "比較動画は短いほうに合わせて切れます。")
EOF

FILTER="[0:v]fps=30,scale=-2:720,setsar=1[a];[1:v]fps=30,scale=-2:720,setsar=1[b];[2:v]fps=30,scale=-2:720,setsar=1[c];[a][b][c]hstack=inputs=3:shortest=1[v]"

if [ -n "$AUDIO" ]; then
  ffmpeg -y -v error -ss "$START" -i "$HUMAN" -i "$RETARGET" -i "$POLICY" -ss "$START" -i "$AUDIO" \
    -filter_complex "$FILTER" -map "[v]" -map 3:a? -c:v libx264 -crf 20 -pix_fmt yuv420p \
    -c:a aac -shortest "$OUT"
else
  ffmpeg -y -v error -ss "$START" -i "$HUMAN" -i "$RETARGET" -i "$POLICY" \
    -filter_complex "$FILTER" -map "[v]" -c:v libx264 -crf 20 -pix_fmt yuv420p "$OUT"
fi
echo "[done] $OUT ($(dur "$OUT") 秒)"
