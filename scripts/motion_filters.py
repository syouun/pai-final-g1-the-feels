"""お手本（G1 の qpos 列）を整えるための関数群（numpy / scipy のみ。MuJoCo 不要）。

qpos の並び: [root xyz(3), root quat wxyz(4), 関節角(29)]

  - detect_outliers / interpolate_bad : 推定ミスによる「1〜数フレームの飛び」を検出して補間で置き換える
  - filter_qpos                        : ゼロ位相 Butterworth 低域通過フィルタ（位相遅れなし＝音ハメが崩れない）
  - moving_average_qpos                : 旧版の移動平均（比較実験用に残す）
  - ground_offsets                     : 接地区間ごとに根の高さを補正する量を求める
  - foot_slide                         : 接地中の足の水平速度（足滑りの指標）
  - per_second                         : フレームごとのフラグを秒ごとに集計する
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# ---------------------------------------------------------------- 基本
def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def align_quat_signs(q: np.ndarray) -> np.ndarray:
    """q と -q は同じ回転。隣のフレームと内積が負なら符号を反転して連続にする。"""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def lowpass(x: np.ndarray, fps: float, cutoff: float, order: int = 4) -> np.ndarray:
    """時間方向 (axis=0) のゼロ位相 Butterworth 低域通過フィルタ。cutoff<=0 なら何もしない。"""
    if cutoff is None or cutoff <= 0 or cutoff >= fps / 2:
        return x
    b, a = butter(order, cutoff / (fps / 2))
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    if padlen < 1:
        return x
    return filtfilt(b, a, x, axis=0, padlen=padlen)


# ---------------------------------------------------------------- 外れ値
def velocities(qpos: np.ndarray, fps: float):
    """フレーム間の速度（長さ T-1）: 根の並進速度 [m/s]、根の角速度 [rad/s]、関節角速度 [rad/s]。"""
    lin = np.linalg.norm(np.diff(qpos[:, 0:3], axis=0), axis=1) * fps
    rot = R.from_quat(wxyz_to_xyzw(qpos[:, 3:7]))
    ang = (rot[1:] * rot[:-1].inv()).magnitude() * fps
    jv = np.abs(np.diff(qpos[:, 7:], axis=0)) * fps
    return lin, ang, jv


def midpoint_residuals(qpos: np.ndarray):
    """各フレームが「前後のフレームの中間」からどれだけ外れているか（長さ T、両端は 0）。

    速いけれど滑らかな動きでは小さく、1フレームだけ姿勢が飛ぶ推定ミスでは大きくなる。
    （速度のしきい値だと、本物の速い振付まで誤検出してしまうため、この方式にした）
    返り値: 根の位置 [m]、根の向き [rad]、関節角の最大値 [rad]
    """
    T = len(qpos)
    pos = np.zeros(T)
    rot = np.zeros(T)
    jnt = np.zeros(T)
    pos[1:-1] = np.linalg.norm(qpos[1:-1, 0:3] - (qpos[:-2, 0:3] + qpos[2:, 0:3]) / 2, axis=1)
    r = R.from_quat(wxyz_to_xyzw(qpos[:, 3:7]))
    half = R.from_rotvec(0.5 * (r[2:] * r[:-2].inv()).as_rotvec())
    rot[1:-1] = (r[1:-1] * (half * r[:-2]).inv()).magnitude()
    jnt[1:-1] = np.abs(qpos[1:-1, 7:] - (qpos[:-2, 7:] + qpos[2:, 7:]) / 2).max(axis=1)
    return pos, rot, jnt


def detect_outliers(qpos: np.ndarray, fps: float, max_pos_jump: float = 0.05,
                    max_rot_jump_deg: float = 30.0, max_joint_jump: float = 0.8,
                    dilate: int = 1) -> np.ndarray:
    """「前後の中間」から大きく外れたフレームを推定ミスとみなして印を付ける。

    しきい値は、同梱のモーションキャプチャ（LAFAN1 ダンス、推定ノイズなし）で
    誤検出が 0 になる値の 1.5〜6 倍にしてある（位置 0.009 m、向き 5°、関節 0.53 rad が最大だった）。
    """
    del fps
    pos, rot, jnt = midpoint_residuals(qpos)
    bad = (pos > max_pos_jump) | (rot > np.radians(max_rot_jump_deg)) | (jnt > max_joint_jump)
    for _ in range(dilate):
        bad[1:] |= bad[:-1].copy()
        bad[:-1] |= bad[1:].copy()
    return bad


def interpolate_bad(qpos: np.ndarray, bad: np.ndarray, fps: float) -> np.ndarray:
    """印の付いたフレームを、前後の正常なフレームから補間で作り直す（姿勢は球面線形補間）。"""
    if not bad.any():
        return qpos
    good = np.where(~bad)[0]
    if len(good) < 2:
        raise RuntimeError("正常なフレームが少なすぎます。推定結果そのものを見直してください。")
    t = np.arange(len(qpos)) / fps
    out = qpos.copy()
    for j in list(range(0, 3)) + list(range(7, qpos.shape[1])):
        out[:, j] = np.interp(t, t[good], qpos[good, j])
    rot_good = R.from_quat(wxyz_to_xyzw(align_quat_signs(qpos[good, 3:7])))
    tq = np.clip(t, t[good[0]], t[good[-1]])
    out[:, 3:7] = xyzw_to_wxyz(Slerp(t[good], rot_good)(tq).as_quat())
    return out


# ---------------------------------------------------------------- 平滑化
def filter_qpos(qpos: np.ndarray, fps: float, cut_root_xy: float = 4.0, cut_root_z: float = 4.0,
                cut_rot: float = 5.0, cut_joint: float = 7.0) -> np.ndarray:
    """部位ごとに別のカットオフで低域通過フィルタをかける。

    120 BPM の拍は 2 Hz。振付のキレ（アクセント）は 6〜8 Hz 程度まで残せば失われにくい。
    推定ノイズ（細かい震え）はそれより高い周波数なので、ここで落とせる。
    """
    out = qpos.copy()
    out[:, 0:2] = lowpass(qpos[:, 0:2], fps, cut_root_xy)
    out[:, 2] = lowpass(qpos[:, 2:3], fps, cut_root_z)[:, 0]
    q = lowpass(align_quat_signs(qpos[:, 3:7]), fps, cut_rot)
    out[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    out[:, 7:] = lowpass(qpos[:, 7:], fps, cut_joint)
    return out


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """旧版の移動平均（端は端の値で埋める）。比較実験用。"""
    if w <= 1:
        return x
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x, np.repeat(x[-1:], pad, axis=0)], axis=0)
    k = np.ones(w) / w
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], axis=1)


def moving_average_qpos(qpos: np.ndarray, w: int) -> np.ndarray:
    out = qpos.copy()
    out[:, 0:3] = moving_average(qpos[:, 0:3], w)
    q = moving_average(align_quat_signs(qpos[:, 3:7]), w)
    out[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    out[:, 7:] = moving_average(qpos[:, 7:], w)
    return out


# ---------------------------------------------------------------- 接地
def foot_speed_xy(foot_pos: np.ndarray, fps: float) -> np.ndarray:
    """foot_pos: (T, 2足, 3) → 足の水平速度 (T, 2) [m/s]（中心差分）。"""
    return np.linalg.norm(np.gradient(foot_pos[:, :, :2], axis=0), axis=2) * fps


def ground_offsets(foot_dist: np.ndarray, foot_v: np.ndarray, fps: float, contact_h: float = 0.03,
                   contact_v: float = 0.4, smooth_hz: float = 1.0, mode: str = "segment"):
    """根の高さに足す補正量 dz (T,) と、接地フラグ (T, 2) を返す。

    foot_dist: (T, 2) 各足と床の最短距離 [m]（負ならめり込み）
    foot_v   : (T, 2) 各足の水平速度 [m/s]

    mode="constant": 旧版と同じ。全体で一律にずらす（10パーセンタイルの足が床に付くように）。
    mode="segment" : まず一律にずらしたあと、「低くて止まっている足」を接地とみなし、
                     接地フレームでは一番低い接地足がちょうど床に付くよう補正量を決める。
                     接地のないフレーム（ジャンプ中など）は前後から補間し、最後に smooth_hz で滑らかにする。
                     → GVHMR の根の高さのドリフト（区間ごとの浮き・めり込み）を打ち消す。
    """
    dmin = foot_dist.min(axis=1)
    s0 = -float(np.percentile(dmin, 10))
    d = foot_dist + s0
    contact = (d < contact_h) & (foot_v < contact_v)
    if mode == "constant":
        return np.full(len(d), s0), contact, {"mode": mode, "constant_shift_m": s0}
    any_c = contact.any(axis=1)
    if any_c.sum() < 2:
        return np.full(len(d), s0), contact, {"mode": "constant(接地フレームが少ないため)", "constant_shift_m": s0}
    lowest = np.where(contact, d, np.inf).min(axis=1)
    idx = np.where(any_c)[0]
    off = np.interp(np.arange(len(d)), idx, -lowest[idx])
    off = lowpass(off[:, None], fps, smooth_hz, order=2)[:, 0]
    info = {"mode": mode, "constant_shift_m": s0,
            "segment_offset_min_m": float(off.min()), "segment_offset_max_m": float(off.max()),
            "contact_ratio": float(any_c.mean())}
    return s0 + off, contact, info


def foot_slide(foot_v: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """接地中の足の水平速度（接地していない足は 0）。(T,)。理想は 0（足は床に止まっているはず）。"""
    return np.where(contact, foot_v, 0.0).max(axis=1)


# ---------------------------------------------------------------- 集計
def per_second(flags: dict[str, np.ndarray], fps: float, t0: float = 0.0) -> list[dict]:
    """フレームごとの True/False を 1 秒ごとに数える。何か 1 つでも立っている秒だけ返す。"""
    T = len(next(iter(flags.values())))
    sec = (np.arange(T) / fps + t0).astype(int)
    rows = []
    for s in np.unique(sec):
        m = sec == s
        row = {"sec": int(s)}
        for k, v in flags.items():
            row[k] = int(v[m].sum())
        if any(row[k] for k in flags):
            rows.append(row)
    return rows


def qpos_from_csv(csv: np.ndarray) -> np.ndarray:
    """BeyondMimic 形式 CSV (xyz, quat xyzw, 関節) → qpos (xyz, quat wxyz, 関節)。"""
    return np.concatenate([csv[:, 0:3], xyzw_to_wxyz(csv[:, 3:7]), csv[:, 7:]], axis=1)


def csv_from_qpos(qpos: np.ndarray) -> np.ndarray:
    return np.concatenate([qpos[:, 0:3], wxyz_to_xyzw(qpos[:, 3:7]), qpos[:, 7:]], axis=1)
