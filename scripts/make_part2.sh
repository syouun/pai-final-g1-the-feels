#!/usr/bin/env bash
# 提出動画のスライドパート（part2.mp4）を作る。【2026-10-19 新規】
#   slide1〜4.png（slides.py の出力）を 10・10・10・9 秒表示し、0.5 秒のクロスフェードでつなぐ（合計 39 秒）。
#   先頭 0.5 秒で明るく、最後の 1 秒で暗く。音声は無音（part1 と同じ AAC 44.1kHz ステレオ、concat でつなぐため）
# 使い方: bash make_part2.sh   （出力: outputs/videos/submission_assets/part2.mp4）
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
A="$HERE/../outputs/videos/submission_assets"
SHOW=${SHOW:-10}            # 1〜3 枚目の表示秒数
LAST=${LAST:-9}             # 4 枚目（最後）の表示秒数【10/19 変更：10 → 9。提出動画を 3 分より 1.2 秒短くするため】
X=0.5                        # クロスフェードの長さ
# 各スライドの入力の長さ：境界の前後 X/2 ずつ重ねる → 表示はちょうど SHOW 秒ずつ
D1=$(python3 -c "print($SHOW+$X/2)"); DM=$(python3 -c "print($SHOW+$X)"); D4=$(python3 -c "print($LAST+$X/2)"); TOTAL=$(python3 -c "print(3*$SHOW+$LAST)")
O1=$(python3 -c "print($SHOW-$X/2)"); O2=$(python3 -c "print(2*$SHOW-$X/2)"); O3=$(python3 -c "print(3*$SHOW-$X/2)")
FO=$(python3 -c "print(3*$SHOW+$LAST-1)")
ffmpeg -nostdin -y -v error \
  -loop 1 -framerate 30 -t $D1 -i "$A/slide1.png" -loop 1 -framerate 30 -t $DM -i "$A/slide2.png" \
  -loop 1 -framerate 30 -t $DM -i "$A/slide3.png" -loop 1 -framerate 30 -t $D4 -i "$A/slide4.png" \
  -f lavfi -t $TOTAL -i anullsrc=channel_layout=stereo:sample_rate=44100 \
  -filter_complex "[0:v]scale=1280:720:flags=lanczos,setsar=1,format=yuv420p[s0];[1:v]scale=1280:720:flags=lanczos,setsar=1,format=yuv420p[s1];\
[2:v]scale=1280:720:flags=lanczos,setsar=1,format=yuv420p[s2];[3:v]scale=1280:720:flags=lanczos,setsar=1,format=yuv420p[s3];\
[s0][s1]xfade=transition=fade:duration=$X:offset=$O1[a];[a][s2]xfade=transition=fade:duration=$X:offset=$O2[b];\
[b][s3]xfade=transition=fade:duration=$X:offset=$O3,fade=t=in:st=0:d=0.5,fade=t=out:st=$FO:d=1,trim=duration=$TOTAL,fps=30[v]" \
  -map "[v]" -map 4:a -c:v libx264 -preset medium -crf 18 -profile:v high -pix_fmt yuv420p -r 30 \
  -c:a aac -b:a 192k -ac 2 -ar 44100 -shortest -movflags +faststart "$A/part2.mp4"
echo "[done] $A/part2.mp4 ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$A/part2.mp4") 秒)"
