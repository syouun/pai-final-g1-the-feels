"""レポート用の図を作る（pai_final フォルダで python3 outputs/report_figs/make_report_figs.py）"""
import re, json, glob, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#9a9993", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
B, O, A, G = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
OUT = "outputs/report_figs"
HARD = [12, 32, 66, 106, 107, 108]

# ---------- 1. 学習曲線（ログから） ----------
keys = {"len": r"Mean episode length:\s*([\d.]+)", "rew": r"Mean reward:\s*([-\d.]+)",
        "bpos": r"error_body_pos:\s*([\d.]+)", "jpos": r"error_joint_pos:\s*([\d.]+)",
        "ee": r"Episode_Termination/ee_body_pos:\s*([\d.]+)", "bin": r"sampling_top1_bin:\s*([\d.]+)"}
txt = open("logs/train_the_feels_full.log", errors="ignore").read()
blocks = re.split(r"Learning iteration\s+(\d+)/\d+", txt)[1:]
rec = {k: [] for k in keys}; its = []
for i in range(0, len(blocks), 2):
    it, body = int(blocks[i]), blocks[i + 1]
    vals = {k: re.search(p, body) for k, p in keys.items()}
    if all(vals.values()):
        its.append(it)
        for k in keys: rec[k].append(float(vals[k].group(1)))
its = np.array(its); rec = {k: np.array(v) for k, v in rec.items()}
sm = lambda x, w=200: np.convolve(x, np.ones(w) / w, mode="valid")
xs = its[199:]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(its, rec["len"] / 50, color=B, alpha=0.18, lw=0.6); ax[0].plot(xs, sm(rec["len"]) / 50, color=B, lw=2)
ax[0].set_title("平均エピソード長 [秒]（上限 10 秒）", loc="left"); ax[0].set_xlabel("反復回数")
ax[1].plot(its, rec["bpos"], color=B, alpha=0.18, lw=0.6); ax[1].plot(xs, sm(rec["bpos"]), color=B, lw=2)
ax[1].set_title("体の位置誤差 [m]（学習中の平均）", loc="left"); ax[1].set_xlabel("反復回数")
ax[2].plot(its, rec["bin"] * 139.84, ".", color=O, ms=1, alpha=0.25)
ax[2].set_title("最も重点的に練習している区間 [秒目]", loc="left"); ax[2].set_xlabel("反復回数"); ax[2].set_ylim(0, 140)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_learning_curve.png", dpi=160); plt.close(fig)

# ---------- 2. チェックポイントごとの評価 ----------
rows = []
for f in glob.glob("outputs/eval/progress_the_feels_full_model_*.json"):
    j = json.load(open(f)); it = int(re.search(r"model_(\d+)", f).group(1))
    rows.append((it, (j["first_fall_s"] or j["duration_s"]), j["mean_joint_err_rad"], j["mean_body_pos_err_rel_m"], j["success"]))
rows.sort(); r = np.array([x[:4] for x in rows], float); ok = np.array([x[4] for x in rows])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].bar(r[:, 0], r[:, 1], width=1200, color=np.where(ok, B, O))
ax[0].set_title("転ばずに踊れた時間 [秒]（全 139.8 秒）", loc="left"); ax[0].set_xlabel("反復回数")
for x, y, s in zip(r[:, 0], r[:, 1], ok):
    if not s: ax[0].text(x, y + 3, f"{y:.1f}s", ha="center", fontsize=8, color="#52514e")
m = ok
ax[1].plot(r[m, 0], r[m, 2], "-o", color=B, lw=2, ms=5); ax[1].set_title("関節角度の誤差 [rad]（完走したモデル）", loc="left"); ax[1].set_xlabel("反復回数")
ax[2].plot(r[m, 0], r[m, 3] * 100, "-o", color=B, lw=2, ms=5); ax[2].set_title("体の位置誤差 [cm]（完走したモデル）", loc="left"); ax[2].set_xlabel("反復回数")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_checkpoints.png", dpi=160); plt.close(fig)

# ---------- 3. 時刻ごとの誤差と、お手本の足の高さ ----------
a = np.loadtxt("outputs/videos/policy_the_feels_full_model_19999.csv", delimiter=",", skiprows=1)
c = np.genfromtxt("outputs/check/the_feels_full_check_frames.csv", delimiter=",", names=True)
foot = np.minimum(c["left_foot_h"], c["right_foot_h"])
roll = lambda x, w: np.convolve(x, np.ones(w) / w, mode="same")
fig, ax = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
ax[0].plot(a[:, 0], a[:, 2] * 100, color=B, lw=0.8); ax[0].axhline(25, color=O, lw=1.2, ls="--")
ax[0].text(1, 23.3, "転倒判定のしきい値（手首・足首の高さのずれ 25 cm）", color="#52514e", fontsize=8)
ax[0].set_title("① 体の位置誤差 [cm]（胴体基準・全リンク平均）", loc="left")
ax[1].plot(a[:, 0], a[:, 1], color=B, lw=0.6, alpha=0.35); ax[1].plot(a[:, 0], roll(a[:, 1], 50), color=B, lw=1.8)
ax[1].set_title("② 関節角度の誤差 [rad]（細線：毎ステップ／太線：1 秒平均）", loc="left")
ax[2].plot(c["time_s"], roll(foot, 15) * 100, color=A, lw=1.4); ax[2].axhline(0, color=G, lw=0.8)
ax[2].set_title("③ お手本で低いほうの足の高さ [cm]（0 なら接地。正なら宙に浮いている）", loc="left"); ax[2].set_xlabel("時刻 [秒]")
for k in ax:
    for h in HARD: k.axvspan(h, h + 1, color=O, alpha=0.10, lw=0)
    k.axvspan(59.815, 80.78, color=G, alpha=0.07, lw=0)
ax[0].text(60.5, 21, "サビ20秒の区間", fontsize=8, color="#52514e")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_error_timeline.png", dpi=160); plt.close(fig)

# ---------- 数値のまとめ ----------
t = a[:, 0]
seg = (t >= 0.0) & (t < 20.94 - 0.0)
s20 = np.loadtxt("outputs/eval/fixcheck_20s_mu0.8.csv", delimiter=",", skiprows=1)
mf = (t >= 59.815) & (t <= 80.78)
summ = {"full_policy_on_20s_segment": {"joint": round(a[mf, 1].mean(), 4), "pos": round(a[mf, 2].mean(), 4)},
        "20s_policy": {"joint": round(s20[:, 1].mean(), 4), "pos": round(s20[:, 2].mean(), 4)},
        "pos_err_0_110s": round(a[t < 110, 2].mean(), 4), "pos_err_110_140s": round(a[t >= 110, 2].mean(), 4),
        "joint_err_0_110s": round(a[t < 110, 1].mean(), 4), "joint_err_110_140s": round(a[t >= 110, 1].mean(), 4),
        "ref_lowest_foot_cm_0_110": round(np.median(foot[c["time_s"] < 110]) * 100, 1),
        "ref_lowest_foot_cm_110_140": round(np.median(foot[c["time_s"] >= 110]) * 100, 1),
        "max_drift_m": round(a[:, 3].max(), 3), "max_drift_t": round(a[a[:, 3].argmax(), 0], 2),
        "first_fullrun_iter_len_ge_9s": int(xs[np.argmax(sm(rec["len"]) / 50 >= 9.0)]) if (sm(rec["len"]) / 50 >= 9.0).any() else None,
        "final_ee_term_per_iter": float(np.mean(rec["ee"][-500:])), "final_top1_bin_sec_median": round(float(np.median(rec["bin"][-2000:])) * 139.84, 1)}
json.dump(summ, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(summ, ensure_ascii=False, indent=1))
