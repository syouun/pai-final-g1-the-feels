#!/usr/bin/env bash
# 提出用動画を作る。【2026-10-19 新規】
#   本編（2分19秒）: 全画面 = ロボット正面 / 左下 = 伊原六花さん踊ってみた / 右下 = TWICE 公式（中抜き同期）
#   その後に説明スライド 4 枚（outputs/videos/submission_assets/part2.mp4）を連結する。
# 使い方: bash make_submission.sh <ロボット動画.mp4> <出力.mp4>
#   ロボット動画は render_policy.py の出力（t=0 が伊原六花さん動画の t=0 と一致していること）
set -euo pipefail
ROBOT=$1; OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)                       # 応用編1の最終課題
A="$HERE/../outputs/videos/submission_assets"
TWICE="$ROOT/TWICE ＂The Feels＂ Choreography Video (Moving Ver.).mp4"
IHARA="$ROOT/【定点】TWICE_The Feels踊ってみた【伊原六花】.mp4"
TMP=$(mktemp -d)
ENC="-c:v libx264 -preset medium -crf 18 -pix_fmt yuv444p -r 30"
# ワイプ素材（ロボット動画に依存しないので一度作ればキャッシュを使う）
# TWICE 公式の切り貼り（音源切り取り分析結果_TWICE_vs_伊原六花.md の通り）
#   0〜6.60 秒: 0:01.58 のコマで静止 / 6.60〜86.99: 公式 1.58〜81.98 / 86.99〜137.10: 公式 146.12〜196.23 / 以降: 最後のコマで静止
if [ ! -s "$A/pip_twice.mp4" ]; then
  ffmpeg -nostdin -y -v warning -i "$TWICE" -filter_complex "[0:v]split=2[t1][t2];\
[t1]trim=start=1.58:end=81.98,setpts=PTS-STARTPTS,fps=30,tpad=start_duration=6.60:start_mode=clone[a];\
[t2]trim=start=146.12:end=196.23,setpts=PTS-STARTPTS,fps=30,tpad=stop_duration=3:stop_mode=clone[b];\
[a][b]concat=n=2:v=1:a=0,scale=400:225:flags=lanczos,setsar=1,trim=duration=139.80[v]" -map "[v]" $ENC -an "$A/pip_twice.mp4"
fi
# 伊原六花さん：先頭からそのまま（音声の相互相関でロボット側とのずれ 0 秒を確認済み）
if [ ! -s "$A/pip_ihara.mp4" ]; then
  ffmpeg -nostdin -y -v warning -i "$IHARA" -vf "fps=30,scale=400:225:flags=lanczos,setsar=1" -t 139.80 $ENC -an "$A/pip_ihara.mp4"
fi
[ "${PIPS_ONLY:-0}" = 1 ] && { echo "[done] ワイプ素材"; exit 0; }
FC="[0:v]fps=30,scale=1280:720,setsar=1[main];\
[main][2:v]overlay=24:471:eof_action=repeat[m0];\
[m0][1:v]overlay=856:471:eof_action=repeat[m1];\
[m1][3:v]overlay=0:0:shortest=1,trim=duration=139.80,fade=t=in:st=0:d=0.4,fade=t=out:st=139.20:d=0.6,format=yuv420p[v];\
[4:a]atrim=duration=139.80,afade=t=out:st=138.6:d=1.2,aresample=44100[aout]"
ffmpeg -nostdin -y -v warning -stats -i "$ROBOT" -i "$A/pip_twice.mp4" -i "$A/pip_ihara.mp4" -loop 1 -framerate 30 -i "$A/overlay2.png" -i "$IHARA" \
  -filter_complex "$FC" -map "[v]" -map "[aout]" -c:v libx264 -preset medium -crf 20 -profile:v high \
  -pix_fmt yuv420p -r 30 -c:a aac -b:a 192k -ac 2 -ar 44100 -movflags +faststart "$TMP/part1.mp4"
printf "file '%s'\nfile '%s'\n" "$TMP/part1.mp4" "$A/part2.mp4" > "$TMP/list.txt"
ffmpeg -nostdin -y -v error -f concat -safe 0 -i "$TMP/list.txt" -c copy -movflags +faststart "$OUT"
echo "[done] $OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT") 秒)"
