#!/usr/bin/env python3
"""お手本 (CSV) が物理的に成り立つかを、学習前に機械的にチェックする。【新規】

強化学習で「何度やっても同じ場所で転ぶ」原因の多くは、お手本そのものが物理的に無理なこと。
このスクリプトは学習に使う G1 のモデル（mjlab 同梱）で、フレームごとに次を調べる。

  1. 逆動力学 (mj_inverse): お手本どおりに動くために体の外から必要な力 F（＝床反力が担うべき力）
       - hover : 両足が浮いているのに上向きの力が必要（空中で止まる／ゆっくり降りる＝重力を無視）
       - pull  : 下向きの力が必要（床は押すことしかできない）
       - slip  : 接地中に必要な水平力が摩擦の限界（μ × 鉛直力）を超える（足が滑る）
  2. 自己干渉 : 腕と胴体などが 5mm 以上めり込む（学習では自己接触に大きな罰がある）
  3. 関節の可動域外 / 関節速度がモーターの上限を超える

結果は秒ごとに集計し、「どの秒が難所か」を一覧・グラフ・JSON で出す。
※ 関節トルクの上限チェックは、床反力を左右の足にどう配分するかが決まらないため行っていない。

使い方（unitree_rl_mjlab の venv、または mujoco が入った環境で）:
  python check_reference.py --csv .../the_feels_20s.csv --out_prefix .../the_feels_20s_check --t0 59.815
"""

import argparse
import json
import re
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_filters as mf  # noqa: E402

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

# G1 のモーターの速度上限 [rad/s]（mjlab の g1_constants.py より）
VEL_LIMITS = [
    (r".*_hip_roll_joint|.*_knee_joint", 20.0),
    (r".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint", 32.0),
    (r".*_wrist_pitch_joint|.*_wrist_yaw_joint", 22.0),
    (r".*", 37.0),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="お手本の CSV（BeyondMimic 形式、36列）")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--t0", type=float, default=0.0, help="表示用：この CSV の先頭が元動画の何秒か")
    p.add_argument("--robot_xml", default=None, help="G1 の xml（省略時は mjlab 同梱の g1.xml）")
    p.add_argument("--out_prefix", required=True, help="出力ファイル名の前半（_check.json / _check.png を付ける）")
    p.add_argument("--mu", type=float, default=0.6, help="足と床の摩擦係数（学習時の既定値 0.6）")
    p.add_argument("--contact_h", type=float, default=0.02, help="接地とみなす足の高さ [m]")
    p.add_argument("--acc_cutoff", type=float, default=10.0, help="加速度計算前の速度の平滑化 [Hz]")
    return p.parse_args()


def load_model(xml):
    spec = mj.MjSpec.from_file(str(xml))
    if not any(g.name == "floor" for g in spec.geoms):
        spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
    return spec.compile()


def default_xml():
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    return G1_XML


def main():
    a = parse_args()
    m = load_model(a.robot_xml or default_xml())
    d = mj.MjData(m)
    qpos = mf.qpos_from_csv(np.loadtxt(a.csv, delimiter=","))
    T, fps, dt = len(qpos), a.fps, 1.0 / a.fps
    assert m.nq == qpos.shape[1], f"モデルの nq={m.nq} と CSV の列 {qpos.shape[1]} が合いません"
    jnames = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    mass = float(m.body_mass.sum())
    mg = mass * 9.81

    # --- 速度・加速度（有限差分。ノイズが増幅されるので速度を少し平滑化してから微分）
    qvel = np.zeros((T, m.nv))
    for i in range(1, T):
        mj.mj_differentiatePos(m, qvel[i], dt, qpos[i - 1], qpos[i])
    qvel[0] = qvel[1]
    qvel = mf.lowpass(qvel, fps, a.acc_cutoff)
    qacc = np.gradient(qvel, dt, axis=0)

    floor = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
    feet = []
    for b in FOOT_BODIES:
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b)
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        feet.append([g for g in gs if m.geom_contype[g] or m.geom_conaffinity[g]] or gs)
    fromto = np.zeros(6)

    F = np.zeros((T, 3))
    foot_d = np.zeros((T, 2))
    selfpen = np.zeros(T)
    selfpair = [""] * T
    flags0 = m.opt.disableflags
    for i in range(T):
        # (1) 逆動力学：拘束（接触・可動域）を切って「外から必要な力」を出す
        m.opt.disableflags = flags0 | mj.mjtDisableBit.mjDSBL_CONSTRAINT
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos[i], qvel[i], qacc[i]
        mj.mj_inverse(m, d)
        F[i] = d.qfrc_inverse[0:3]  # 浮遊ベースの並進成分（ワールド座標）
        # (2) 接触を戻して、足の高さと自己干渉
        m.opt.disableflags = flags0
        mj.mj_forward(m, d)
        for k in range(2):
            foot_d[i, k] = min(mj.mj_geomDistance(m, d, g, floor, 1.0, fromto) for g in feet[k])
        worst = 0.0
        for j in range(d.ncon):
            c = d.contact[j]
            if c.geom1 == floor or c.geom2 == floor:
                continue
            if c.dist < worst:
                worst = c.dist
                selfpair[i] = (f"{mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1])}"
                               f" × {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])}")
        selfpen[i] = -worst
    m.opt.disableflags = flags0

    fz = F[:, 2] / mg
    fxy = np.linalg.norm(F[:, :2], axis=1) / mg
    contact = foot_d.min(1) < a.contact_h
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    q = qpos[:, 7:]
    jl_viol = ((q < lo - 0.02) | (q > hi + 0.02)) & (m.jnt_limited[1:] > 0)
    vlim = np.array([next(v for pat, v in VEL_LIMITS if re.fullmatch(pat, n)) for n in jnames])
    vel_viol = np.abs(qvel[:, 6:]) > vlim

    flags = {
        "hover": (~contact) & (fz > 0.3),
        "pull": fz < -0.2,
        "slip": contact & (fz > 0.1) & (fxy > a.mu * fz),
        "self_collision": selfpen > 0.005,
        "joint_limit": jl_viol.any(1),
        "joint_vel": vel_viol.any(1),
    }
    rows = mf.per_second(flags, fps, t0=a.t0)
    total = {k: int(v.sum()) for k, v in flags.items()}

    def top_joints(viol):
        cnt = viol.sum(0)
        return {jnames[j]: int(cnt[j]) for j in np.argsort(-cnt)[:5] if cnt[j] > 0}

    pairs = {}
    for i in np.where(flags["self_collision"])[0]:
        pairs[selfpair[i]] = pairs.get(selfpair[i], 0) + 1

    summary = {
        "csv": a.csv, "frames": T, "duration_s": round(T / fps, 2), "t0": a.t0, "mass_kg": round(mass, 2),
        "frames_flagged": total,
        "frames_any_flag": int(np.any(np.stack(list(flags.values())), axis=0).sum()),
        "airborne_ratio": round(float((~contact).mean()), 3),
        "peak_required_vertical_force_mg": round(float(fz.max()), 2),
        "joint_limit_top": top_joints(jl_viol),
        "joint_vel_top": top_joints(vel_viol),
        "self_collision_pairs": dict(sorted(pairs.items(), key=lambda x: -x[1])[:8]),
        "per_second": rows,
    }
    out = Path(a.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + "_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    np.savetxt(str(out) + "_check_frames.csv",
               np.column_stack([np.arange(T) / fps + a.t0, fz, fxy, foot_d, selfpen] +
                               [v.astype(int) for v in flags.values()]),
               delimiter=",", fmt="%.4f", comments="",
               header="time_s,Fz_per_mg,Fxy_per_mg,left_foot_h,right_foot_h,self_pen_m," + ",".join(flags))

    print(f"[check] {T} フレーム ({T / fps:.1f} 秒)  体重 {mass:.1f} kg")
    print("[check] 問題のあるフレーム数: " + ", ".join(f"{k} {v}" for k, v in total.items()))
    if summary["self_collision_pairs"]:
        print(f"[check] 自己干渉の組: {summary['self_collision_pairs']}")
    if summary["joint_vel_top"]:
        print(f"[check] 速度上限を超える関節: {summary['joint_vel_top']}")
    print("[check] 難所（問題が 3 フレーム以上ある秒）:")
    for r in rows:
        n = sum(v for k, v in r.items() if k != "sec")
        if n >= 3:
            print("   ", r["sec"], "秒:", {k: v for k, v in r.items() if k != "sec" and v})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(T) / fps + a.t0
        fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
        ax[0].plot(t, fz, lw=1, label="required vertical force / mg")
        ax[0].plot(t, fxy, lw=1, label="required horizontal force / mg")
        ax[0].fill_between(t, -1, 4, where=~contact, color="0.85", label="both feet airborne")
        ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_ylim(-1, 4)
        ax[0].legend(loc="upper right", fontsize=8)
        for k, (name, v) in enumerate(flags.items()):
            ax[1].scatter(t[v], np.full(v.sum(), k), s=4)
        ax[1].set_yticks(range(len(flags)), list(flags))
        ax[1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(str(out) + "_check.png", dpi=110)
        print(f"[check] グラフ: {out}_check.png")
    except Exception as e:  # matplotlib がない・日本語フォントがない等
        print(f"[check] グラフは省略しました（{type(e).__name__}）")
    print(f"[check] 詳細: {out}_check.json / {out}_check_frames.csv")


if __name__ == "__main__":
    main()
