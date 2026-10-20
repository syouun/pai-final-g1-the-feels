#!/usr/bin/env python3
"""提出レポート用の図（2026-10-19）。pai_final/ で実行: python3 outputs/report_figs/make_final_figs.py"""
import json, subprocess, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    fm.fontManager.addfont(p)
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
OUT = 'outputs/report_figs/final/'
C_OLD, C_MID, C_NEW, C_ACC = '#9aa3b5', '#5ec8ff', '#ff6f91', '#2b3a55'

# ---- 図A：段階ごとの誤差（摩擦 0.8、全編）
st = [('NB2\nゼロから学習', 0.1635, 9.32, 22.65, 31.2), ('NB3\n浮きの補正＋追加学習', 0.1030, 3.90, 16.70, 23.9),
      ('NB4\nしゃがみ補正＋追加学習', 0.0943, 3.50, 9.37, 16.8), ('提出\nmodel_26500', 0.0934, 3.50, 9.08, 18.5)]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
cols = [C_OLD, C_MID, C_MID, C_NEW]
for a, (k, t, u) in zip(ax, [(1, '関節角度誤差（平均）', 'rad'), (2, '位置誤差（平均）', 'cm'), (3, '最大位置誤差', 'cm')]):
    v = [s[k] for s in st]; b = a.bar(range(4), v, color=cols)
    for i, x in enumerate(v): a.text(i, x, f'{x:g}', ha='center', va='bottom', fontsize=10)
    a.set_xticks(range(4)); a.set_xticklabels([s[0] for s in st], fontsize=8); a.set_title(f'{t} [{u}]', fontsize=11)
    a.spines[['top', 'right']].set_visible(False); a.set_ylim(0, max(v) * 1.18)
fig.suptitle('段階ごとの追従誤差（摩擦 0.8、全編 139.8 秒）', fontsize=12)
plt.tight_layout(); plt.savefig(OUT + 'figA_progress.png', dpi=170); plt.close()

# ---- 図B：摩擦への強さ（完走の表）
rows = ['NB2 (model_19999)', 'NB3 (model_23998)', 'NB4 (model_26997)', 'NB5 (model_28500)', '提出 (model_26500)']
mus = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2]
# 1 = 完走、0 = 停止、nan = 未評価
M = np.array([[1, 1, np.nan, 1, np.nan, np.nan], [1, 1, np.nan, 1, np.nan, np.nan], [1, 1, 1, 0, 1, 0],
              [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], float)
lab = [['○', '○', '−', '○', '−', '−'], ['○', '○', '−', '○', '−', '−'], ['○', '○', '○', '×\n(2/7回停止)', '○', '×'],
       ['○', '○', '○', '○', '○', '×'], ['○', '○', '○', '○\n(6/6回)', '○', '○']]
fig, a = plt.subplots(figsize=(9, 3.4))
cm = matplotlib.colors.ListedColormap(['#ffd0da', '#d8f3e6'])
a.imshow(np.nan_to_num(M, nan=0.5) , cmap=matplotlib.colors.ListedColormap(['#ffd0da', '#eeeeee', '#d8f3e6']), vmin=0, vmax=1, aspect='auto')
for i in range(len(rows)):
    for j in range(len(mus)):
        a.text(j, i, lab[i][j], ha='center', va='center', fontsize=10, color=C_ACC)
a.set_xticks(range(len(mus))); a.set_xticklabels([f'摩擦 {m}' for m in mus]); a.set_yticks(range(len(rows))); a.set_yticklabels(rows)
a.set_title('足裏の摩擦を変えたときの完走（○ 完走 / × 途中で停止 / − 未評価）', fontsize=11)
for s in a.spines.values(): s.set_visible(False)
plt.tight_layout(); plt.savefig(OUT + 'figB_friction.png', dpi=170); plt.close()

# ---- 図C：時刻ごとの誤差（NB2 と提出、1 秒平均）
def per_sec(f):
    d = pd.read_csv(f); d['s'] = d.time_s.astype(int); return d.groupby('s').agg(j=('joint_err_rad', 'mean'), p=('body_pos_err_rel_m', 'mean'))
a0 = per_sec('outputs/eval/cmp_old_policy_old_ref.csv'); a1 = per_sec('outputs/videos/front_policy_final_ft2_model_26500_noghost.csv')
fig, ax = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True)
ax[0].plot(a0.index, a0.j, color=C_OLD, label='NB2 ゼロから学習（元のお手本）'); ax[0].plot(a1.index, a1.j, color=C_NEW, label='提出 model_26500')
ax[1].plot(a0.index, a0.p * 100, color=C_OLD); ax[1].plot(a1.index, a1.p * 100, color=C_NEW)
ax[0].set_ylabel('関節角度誤差 [rad]'); ax[1].set_ylabel('位置誤差 [cm]'); ax[1].set_xlabel('時刻 [秒]'); ax[0].legend(fontsize=9)
for a in ax:
    a.axvspan(9, 14, color='#ffe9a8', alpha=0.6); a.axvspan(110, 139.8, color='#e3f2ff', alpha=0.6); a.spines[['top', 'right']].set_visible(False)
ax[0].text(11.5, ax[0].get_ylim()[1] * 0.92, 'しゃがみ', ha='center', fontsize=9); ax[0].text(125, ax[0].get_ylim()[1] * 0.92, '終盤（お手本の浮きを補正）', ha='center', fontsize=9)
fig.suptitle('時刻ごとの追従誤差（1 秒平均、摩擦 0.8）', fontsize=12)
plt.tight_layout(); plt.savefig(OUT + 'figC_timeline.png', dpi=170); plt.close()
print('NB2 per-sec max jnt %.3f, final %.3f' % (a0.j.max(), a1.j.max()))

# ---- 図D：3 画面のコマ（人／お手本／ロボット）
V = 'outputs/videos/comparison_final_ft2_model_26500_front_all_noghost.mp4'
ts = [20, 70, 121]
fr = []
for t in ts:
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(t), '-i', V, '-frames:v', '1', '-vf', 'scale=1920:-2', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    fr.append(np.frombuffer(raw, np.uint8).reshape(-1, 1920, 3))
fig, ax = plt.subplots(len(ts), 1, figsize=(12, 3.25 * len(ts)))
for a, im, t in zip(ax, fr, ts):
    a.imshow(im); a.axis('off'); a.set_title(f'{t} 秒　左：人の 3D 推定（GVHMR）／中央：お手本（G1 へのリターゲット）／右：物理シミュレーションの G1（方策）', fontsize=9)
plt.tight_layout(); plt.savefig(OUT + 'figD_frames.png', dpi=130); plt.close()
print('ok')
