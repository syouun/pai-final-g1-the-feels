#!/usr/bin/env python3
"""お手本 (CSV) の一部の区間だけを、物理的に無理のない動きに直す。【2026-10-18 新規】

対象は 10〜12 秒の「深いしゃがみ」のような区間。次の 3 つを、区間の前後で滑らかにつながるように行う。

  1. しゃがみを浅くする（--depth_keep）
       脚の曲げ（股関節の前後・膝・足首の前後）を、区間の直前に立っていたときの角度へ近づける。
       q' = q_立ち + (1 - (1 - depth_keep) * w(t)) * (q - q_立ち)   w(t): 区間で 1、前後 --ramp 秒で 0 へ（余弦）
       depth_keep = 1.0 なら変更なし、0.6 なら曲げを 4 割浅くする。
  2. 腰の高さを合わせる
       脚を伸ばすと体が床にめり込むので、「低いほうの足と床の距離」が元のお手本と同じになるよう根の高さを上げる。
  3. 腕のめり込みを外へ逃がす（--arm_fix）
       手首と体・手首同士がめり込むフレームで、その腕の肩の横開き（shoulder_roll）を、めり込みが消える最小量だけ外へ開く。
       開く向きは区間全体でめり込みが減るほうを自動で選び、フレーム間で震えないように平滑化する。

腕・脚以外（根の水平位置・向き、腰・腕の他の関節）は変えない。結果と前後比較を JSON に保存する。

使い方（mujoco と scipy が入った環境。unitree_rl_mjlab の venv で可）:
  python edit_reference.py --csv .../the_feels_full_g25.csv --out_csv .../the_feels_full_g25s.csv \
      --start 9.5 --end 13.5 --depth_keep 0.6
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
LEG_PAT = r"(left|right)_(hip_pitch|knee|ankle_pitch)_joint"
ARM_PAT = r"(left|right)_(shoulder|elbow|wrist|hand|rubber).*"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start", type=float, required=True, help="直す区間の開始 [秒]")
    p.add_argument("--end", type=float, required=True, help="直す区間の終了 [秒]")
    p.add_argument("--ramp", type=float, default=0.5, help="区間の前後でなめらかにつなぐ時間 [秒]")
    p.add_argument("--depth_keep", type=float, default=0.6, help="脚の曲げを何割残すか（1.0=変更なし）")
    p.add_argument("--stand_window", type=float, default=2.0, help="区間の直前、何秒を「立っている角度」の基準にするか")
    p.add_argument("--no_arm_fix", action="store_true", help="腕のめり込み補正をしない")
    p.add_argument("--max_roll", type=float, default=0.6, help="肩を外へ開く最大量 [rad]")
    p.add_argument("--pen_tol", type=float, default=0.005, help="許すめり込み [m]")
    p.add_argument("--robot_xml", default=None, help="G1 の xml（省略時は mjlab 同梱。腕の当たり判定がある版を使うこと）")
    return p.parse_args()


def load_model(xml):
    spec = mj.MjSpec.from_file(str(xml))
    if not any(g.name == "floor" for g in spec.geoms):
        spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
    return spec.compile()


def default_xml():
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    return G1_XML


class Robot:
    def __init__(self, m):
        self.m, self.d = m, mj.MjData(m)
        self.floor = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
        self.jnames = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]  # qpos[7:] の順
        self.feet = []
        for b in FOOT_BODIES:
            bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b)
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
            self.feet.append([g for g in gs if m.geom_contype[g] or m.geom_conaffinity[g]] or gs)
        self.bname = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "" for g in range(m.ngeom)]
        self.ft = np.zeros(6)

    def set(self, q):
        self.d.qpos[:] = q
        mj.mj_forward(self.m, self.d)

    def foot_min(self, q):
        self.set(q)
        return min(mj.mj_geomDistance(self.m, self.d, g, self.floor, 1.0, self.ft) for f in self.feet for g in f)

    def pen(self, q):
        """自己干渉: (全体の最大めり込み, {'left': 左腕が絡む最大めり込み, 'right': ...}, 最悪の組)"""
        self.set(q)
        worst, side, pair = 0.0, {"left": 0.0, "right": 0.0}, ""
        for j in range(self.d.ncon):
            c = self.d.contact[j]
            if c.geom1 == self.floor or c.geom2 == self.floor or c.dist >= 0:
                continue
            b1, b2 = self.bname[c.geom1], self.bname[c.geom2]
            if -c.dist > worst:
                worst, pair = -c.dist, f"{b1} × {b2}"
            for b in (b1, b2):
                mt = re.fullmatch(ARM_PAT, b)
                if mt:
                    side[mt.group(1)] = max(side[mt.group(1)], -c.dist)
        return worst, side, pair


def main():
    a = parse_args()
    fps = a.fps
    csv = np.loadtxt(a.csv, delimiter=",")
    q0 = mf.qpos_from_csv(csv)
    T = len(q0)
    R = Robot(load_model(a.robot_xml or default_xml()))
    assert R.m.nq == q0.shape[1], f"nq={R.m.nq} と CSV の列 {q0.shape[1]} が不一致"
    t = np.arange(T) / fps

    # 重み w(t)：区間で 1、前後 ramp 秒で余弦に 0 へ
    w = np.zeros(T)
    w[(t >= a.start) & (t <= a.end)] = 1.0
    up = (t >= a.start - a.ramp) & (t < a.start)
    dn = (t > a.end) & (t <= a.end + a.ramp)
    w[up] = 0.5 - 0.5 * np.cos(np.pi * (t[up] - (a.start - a.ramp)) / a.ramp)
    w[dn] = 0.5 + 0.5 * np.cos(np.pi * (t[dn] - a.end) / a.ramp)
    idx = np.where(w > 0)[0]
    print(f"[edit] 区間 {a.start}〜{a.end} 秒（前後 {a.ramp} 秒でつなぐ）: {len(idx)} フレーム")

    # 変更前の状態
    before = {"foot_min": np.array([R.foot_min(q0[i]) for i in idx]), "pen": [R.pen(q0[i]) for i in idx]}

    # 1. しゃがみを浅く
    q = q0.copy()
    leg = [k for k, n in enumerate(R.jnames) if re.fullmatch(LEG_PAT, n)]
    s0 = max(0, int((a.start - a.ramp - a.stand_window) * fps))
    s1 = max(s0 + 1, int((a.start - a.ramp) * fps))
    q_stand = np.median(q0[s0:s1, 7:], axis=0)
    keep = 1.0 - (1.0 - a.depth_keep) * w
    for k in leg:
        q[:, 7 + k] = q_stand[k] + keep * (q0[:, 7 + k] - q_stand[k])
    print("[edit] 脚の関節: " + ", ".join(R.jnames[k] for k in leg))

    # 2. 腰の高さ：低いほうの足と床の距離を元と同じにする
    dz = np.zeros(T)
    for n, i in enumerate(idx):
        dz[i] = before["foot_min"][n] - R.foot_min(q[i])
    dz = mf.lowpass(dz[:, None], fps, 3.0, order=2)[:, 0] * (w > 0)
    q[:, 2] += dz
    print(f"[edit] 腰を上げた量: 最大 {dz.max() * 100:.1f} cm")

    # 3. 腕のめり込みを外へ逃がす
    roll_delta = {"left": np.zeros(T), "right": np.zeros(T)}
    if not a.no_arm_fix:
        roll = {s: R.jnames.index(f"{s}_shoulder_roll_joint") for s in ("left", "right")}
        need = {s: [i for i in idx if R.pen(q[i])[1][s] > a.pen_tol] for s in ("left", "right")}
        for s in ("left", "right"):
            if not need[s]:
                continue
            # 開く向き：区間全体でめり込みの合計が減るほう
            score = {}
            for sign in (+1, -1):
                tot = 0.0
                for i in need[s]:
                    qq = q[i].copy()
                    qq[7 + roll[s]] += sign * 0.25
                    tot += R.pen(qq)[1][s]
                score[sign] = tot
            sign = min(score, key=score.get)
            steps = np.arange(0.025, a.max_roll + 1e-9, 0.025)
            for i in need[s]:
                for dlt in steps:
                    qq = q[i].copy()
                    qq[7 + roll[s]] += sign * dlt
                    if R.pen(qq)[1][s] <= a.pen_tol:
                        break
                roll_delta[s][i] = sign * dlt
            # 震えないように：前後 0.2 秒の最大値で広げてから平滑化し、区間の重みをかける
            r = roll_delta[s]
            k = int(0.2 * fps)
            mag = np.array([np.abs(r[max(0, i - k):i + k + 1]).max() for i in range(T)])
            r = sign * mf.lowpass(mag[:, None], fps, 2.0, order=2)[:, 0].clip(0) * np.minimum(1.0, w * 2)
            roll_delta[s] = r
            q[:, 7 + roll[s]] += r
            lo, hi = R.m.jnt_range[1 + roll[s]]
            q[:, 7 + roll[s]] = np.clip(q[:, 7 + roll[s]], lo, hi)
            print(f"[edit] {s} 腕: めり込み {len(need[s])} フレーム → 肩を {'+' if sign > 0 else '-'} 向きに最大 "
                  f"{np.abs(r).max():.2f} rad 開いた")

    # 変更後の状態と保存
    after = {"foot_min": np.array([R.foot_min(q[i]) for i in idx]), "pen": [R.pen(q[i]) for i in idx]}
    pen_b = np.array([p[0] for p in before["pen"]])
    pen_a = np.array([p[0] for p in after["pen"]])
    pairs_b, pairs_a = {}, {}
    for (pb, _, nb), (pa, _, na) in zip(before["pen"], after["pen"]):
        if pb > a.pen_tol:
            pairs_b[nb] = pairs_b.get(nb, 0) + 1
        if pa > a.pen_tol:
            pairs_a[na] = pairs_a.get(na, 0) + 1
    rep = {
        "source_csv": a.csv, "start_s": a.start, "end_s": a.end, "ramp_s": a.ramp, "depth_keep": a.depth_keep,
        "frames_edited": int(len(idx)),
        "pelvis_min_height_m": {"before": round(float(q0[idx, 2].min()), 3), "after": round(float(q[idx, 2].min()), 3)},
        "max_root_raise_cm": round(float(dz.max() * 100), 1),
        "foot_min_dist_m": {"before_min": round(float(before["foot_min"].min()), 3), "after_min": round(float(after["foot_min"].min()), 3),
                            "max_abs_change": round(float(np.abs(after["foot_min"] - before["foot_min"]).max()), 3)},
        "self_penetration_m": {"before_max": round(float(pen_b.max()), 3), "after_max": round(float(pen_a.max()), 3),
                               "frames_over_tol_before": int((pen_b > a.pen_tol).sum()), "frames_over_tol_after": int((pen_a > a.pen_tol).sum())},
        "self_collision_pairs": {"before": pairs_b, "after": pairs_a},
        "shoulder_roll_delta_max_rad": {s: round(float(np.abs(v).max()), 3) for s, v in roll_delta.items()},
        "leg_joint_change_max_rad": round(float(np.abs(q[:, 7:] - q0[:, 7:])[:, leg].max()), 3),
    }
    out = Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, mf.csv_from_qpos(q).astype(np.float32), delimiter=",", fmt="%.6f")
    out.with_name(out.stem + "_edit_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
