"""ノートブック3（お手本修正＋追加学習）の比較図（pai_final フォルダで実行）"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#9a9993",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
C = [("old_policy_old_ref", "① 旧方策 × 旧お手本", "#9a9993"), ("old_policy_new_ref", "② 旧方策 × 新お手本（お手本の修正のみ）", "#eb6834"),
     ("new_policy_new_ref", "③ 新方策 × 新お手本（修正＋追加学習 4,000 反復）", "#2a78d6")]
sm = lambda x, w=50: np.convolve(x, np.ones(w) / w, mode="same")
fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
for key, lab, col in C:
    a = np.loadtxt(f"outputs/eval/cmp_{key}.csv", delimiter=",", skiprows=1)
    ax[0].plot(a[:, 0], sm(a[:, 2]) * 100, color=col, lw=1.6, label=lab)
    ax[1].plot(a[:, 0], sm(a[:, 1]), color=col, lw=1.6, label=lab)
ax[0].set_title("体の位置誤差 [cm]（1 秒平均）", loc="left"); ax[1].set_title("関節角度の誤差 [rad]（1 秒平均）", loc="left")
ax[1].set_xlabel("時刻 [秒]"); ax[0].legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("outputs/report_figs/fig4_finetune_timeline.png", dpi=160); plt.close(fig)
rows = []
for key, lab, _ in C:
    j = json.load(open(f"outputs/eval/cmp_{key}.json"))
    rows.append([lab, j["success"], j["mean_joint_err_rad"], round(j["mean_body_pos_err_rel_m"] * 100, 1), round(j["max_body_pos_err_rel_m"] * 100, 1),
                 round(j["max_root_xy_drift_m"] * 100, 1), j["mean_action_rate"]])
fig, ax = plt.subplots(figsize=(13, 1.9)); ax.axis("off")
t = ax.table(cellText=[[str(v) for v in r] for r in rows], colLabels=["条件（摩擦 0.8）", "完走", "関節誤差 [rad]", "位置誤差 [cm]", "最大位置誤差 [cm]", "最大ドリフト [cm]", "行動の変化量"],
             loc="center", cellLoc="center", colWidths=[0.38, 0.06, 0.1, 0.1, 0.12, 0.12, 0.1])
t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.6)
fig.savefig("outputs/report_figs/table_finetune.png", dpi=160, bbox_inches="tight"); plt.close(fig)
print("ok")
