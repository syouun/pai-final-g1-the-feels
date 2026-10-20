#!/usr/bin/env python3
"""GVHMR の推定結果 (hmr4d_results.pt) を Unitree G1 のお手本 (CSV) に変換する。【改訂版】

GMR 付属の scripts/gvhmr_to_robot.py は画面表示が前提で Colab では動かないため、表示なしで次を行う。

  1. GVHMR の SMPL-X を読み込み、GMR で G1 (29自由度) の関節角へリターゲット（全フレーム）
     → 加工前の結果を <name>_raw.csv に保存（後でフィルタだけやり直せるように）
  2. 推定ミス（1〜数フレームの姿勢の飛び）を検出し、前後から補間して置き換え
  3. ゼロ位相 Butterworth フィルタで平滑化（位相遅れなし。旧版の移動平均も --filter ma で選べる）
  4. --start / --end で区間を切り出す（フィルタは全体にかけてから切るので、端が崩れない）
  5. 接地区間ごとに根の高さを補正（GVHMR の高さのドリフト対策）し、足滑りを測る
  6. CSV（BeyondMimic 形式）、pkl、プレビュー動画、診断レポート (JSON) を保存

すでにある CSV を加工し直すだけなら --from_csv を使う（GMR の IK を解き直さないので速い）。

CSV の各行: root位置 xyz(3) + root姿勢 クォータニオン xyzw(4) + 関節角(29) = 36列

使い方（GMR のフォルダで実行）:
  # 全編を変換
  MUJOCO_GL=egl python gvhmr_to_g1_csv.py --gvhmr_pred_file .../hmr4d_results.pt \
      --out_dir .../motions --name the_feels_full
  # 全編の raw から 20 秒区間を作る
  MUJOCO_GL=egl python gvhmr_to_g1_csv.py --from_csv .../motions/the_feels_full_raw.csv \
      --out_dir .../motions --name the_feels_20s --start 59.815 --end 80.78
"""

import argparse
import json
import os
import pathlib
import pickle
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mujoco as mj
import numpy as np

import motion_filters as mf

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gvhmr_pred_file", help="GVHMR の hmr4d_results.pt")
    src.add_argument("--from_csv", help="既存の CSV（例: *_raw.csv）を加工し直す")
    p.add_argument("--out_dir", required=True, help="出力フォルダ")
    p.add_argument("--name", required=True, help="出力ファイル名（拡張子なし）")
    p.add_argument("--gmr_root", default=None, help="GMR リポジトリのパス（省略時はカレントディレクトリ）")
    p.add_argument("--robot_xml", default=None, help="接地計算・プレビューに使う G1 の xml（省略時は GMR のもの）")
    p.add_argument("--fps", type=float, default=30.0, help="お手本の fps（GVHMR は 30fps 前提。動画は 30fps に変換しておく）")
    p.add_argument("--start", type=float, default=0.0, help="切り出す区間の開始 [秒]")
    p.add_argument("--end", type=float, default=None, help="切り出す区間の終了 [秒]（省略時は最後まで）")
    # 推定ミスの除去
    p.add_argument("--no_outlier", action="store_true", help="推定ミスの検出・補間をしない")
    p.add_argument("--max_pos_jump", type=float, default=0.05, help="根の位置の飛び [m]")
    p.add_argument("--max_rot_jump_deg", type=float, default=30.0, help="根の向きの飛び [deg]")
    p.add_argument("--max_joint_jump", type=float, default=0.8, help="関節角の飛び [rad]")
    # 平滑化
    p.add_argument("--filter", choices=["butter", "ma", "none"], default="butter",
                   help="butter: ゼロ位相 Butterworth（推奨） / ma: 旧版の移動平均 / none: なし")
    p.add_argument("--cut_joint", type=float, default=7.0, help="関節角のカットオフ [Hz]")
    p.add_argument("--cut_root_xy", type=float, default=4.0, help="根の水平位置のカットオフ [Hz]")
    p.add_argument("--cut_root_z", type=float, default=4.0, help="根の高さのカットオフ [Hz]")
    p.add_argument("--cut_rot", type=float, default=5.0, help="根の向きのカットオフ [Hz]")
    p.add_argument("--smooth", type=int, default=5, help="--filter ma のときの窓幅 [フレーム]")
    # 接地
    p.add_argument("--ground", choices=["segment", "constant", "none"], default="segment",
                   help="segment: 接地区間ごとに補正（推奨） / constant: 旧版の一律補正 / none: 補正なし")
    p.add_argument("--contact_h", type=float, default=0.03, help="接地とみなす足の高さ [m]")
    p.add_argument("--contact_v", type=float, default=0.4, help="接地とみなす足の水平速度 [m/s]")
    p.add_argument("--z_offset", type=float, default=0.0, help="追加で根の高さをずらす [m]")
    # 出力
    p.add_argument("--no_video", action="store_true", help="プレビュー動画を書き出さない")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    # 【2026-10-18 追加】プレビュー動画のカメラ（G1 の正面 = +x。カメラは azimuth の方向を向く）
    p.add_argument("--cam_azimuth", type=float, default=90.0, help="カメラの水平角 [deg]（既定 90。お手本の向きが yaw≈90° なので背面）")
    p.add_argument("--cam_elevation", type=float, default=-10.0, help="カメラの仰角 [deg]")
    p.add_argument("--cam_distance", type=float, default=3.0, help="カメラ距離 [m]")
    return p.parse_args()


# ---------------------------------------------------------------- MuJoCo モデル
def load_model(xml_path: str) -> mj.MjModel:
    """xml を読み込む。床 (floor) がなければ追加する（mjlab の G1 には床がないため）。"""
    spec = mj.MjSpec.from_file(str(xml_path))
    if not any(g.name == "floor" for g in spec.geoms):
        spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
    return spec.compile()


def foot_geoms(model):
    """足（ankle_roll_link）の当たり判定用ジオム。衝突が有効なものを優先し、なければ全部。"""
    out = []
    for b in FOOT_BODIES:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b)
        if bid < 0:
            raise RuntimeError(f"{b} がモデルにありません")
        gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
        col = [g for g in gs if model.geom_contype[g] or model.geom_conaffinity[g]]
        out.append(col or gs)
    return out


def foot_state(model, qpos_seq):
    """各フレームの (足と床の最短距離 (T,2), 足の位置 (T,2,3))。"""
    data = mj.MjData(model)
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    geoms = foot_geoms(model)
    bids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
    fromto = np.zeros(6)
    dist = np.empty((len(qpos_seq), 2))
    pos = np.empty((len(qpos_seq), 2, 3))
    for i, q in enumerate(qpos_seq):
        data.qpos[:] = q
        mj.mj_forward(model, data)
        for k in range(2):
            dist[i, k] = min(mj.mj_geomDistance(model, data, g, floor_id, 1.0, fromto) for g in geoms[k])
            pos[i, k] = data.xpos[bids[k]]
    return dist, pos


def render_preview(model, qpos_seq, fps, path, width, height, azimuth=90.0, elevation=-10.0, distance=3.0):
    import imageio
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=height, width=width)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for q in qpos_seq:
        data.qpos[:] = q
        mj.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
    writer.close()
    renderer.close()


# ---------------------------------------------------------------- GVHMR → qpos
def retarget_gvhmr(pred_file, gmr_root, fps):
    import torch
    # PyTorch 2.6 以降は torch.load の既定が weights_only=True で、GVHMR の結果を読めないことがある
    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{"weights_only": False, **k})
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

    smplx_folder = gmr_root / "assets" / "body_models"
    if not (smplx_folder / "smplx" / "SMPLX_NEUTRAL.npz").exists():
        raise FileNotFoundError(f"{smplx_folder}/smplx/SMPLX_NEUTRAL.npz がありません（手順書 2-5 参照）")
    smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(pred_file, smplx_folder)
    frames, out_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=int(fps))
    print(f"[info] 推定身長 {float(human_height):.2f} m / {len(frames)} フレーム ({len(frames) / out_fps:.1f} 秒)")
    retarget = GMR(actual_human_height=human_height, src_human="smplx", tgt_robot="unitree_g1", verbose=False)
    qpos = []
    for i, f in enumerate(frames):
        qpos.append(retarget.retarget(f).copy())  # [root xyz, root quat wxyz, 29 joints]
        if i % 300 == 0:
            print(f"[retarget] {i}/{len(frames)}")
    return np.asarray(qpos), float(human_height)


def default_robot_xml():
    from general_motion_retargeting import ROBOT_XML_DICT
    return str(ROBOT_XML_DICT["unitree_g1"])


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = args.fps
    report = {"name": args.name, "fps": fps, "start_s": args.start, "end_s": args.end}

    # 1. 読み込み
    if args.gvhmr_pred_file:
        qpos, h = retarget_gvhmr(args.gvhmr_pred_file, pathlib.Path(args.gmr_root or os.getcwd()), fps)
        report["human_height_m"] = h
        raw_path = out_dir / f"{args.name}_raw.csv"
        np.savetxt(raw_path, mf.csv_from_qpos(qpos), delimiter=",", fmt="%.6f")
        print(f"[save] 加工前: {raw_path}")
    else:
        qpos = mf.qpos_from_csv(np.loadtxt(args.from_csv, delimiter=","))
        report["source_csv"] = args.from_csv
    print(f"[info] {len(qpos)} フレーム ({len(qpos) / fps:.1f} 秒)")

    # 2. 推定ミスの除去（全体に対して）
    if not args.no_outlier:
        bad = mf.detect_outliers(qpos, fps, args.max_pos_jump, args.max_rot_jump_deg, args.max_joint_jump)
        report["outlier_frames"] = int(bad.sum())
        report["outlier_seconds"] = sorted({round(i / fps, 1) for i in np.where(bad)[0]})
        print(f"[outlier] 推定ミスとみなしたフレーム: {bad.sum()} 個"
              + (f"（時刻 {report['outlier_seconds'][:20]} …）" if bad.any() else ""))
        if bad.mean() > 0.05:
            print("[warn] 5% を超えるフレームが推定ミス扱いです。2_global.mp4 で推定結果そのものを確認してください。")
        qpos = mf.interpolate_bad(qpos, bad, fps)

    # 3. 平滑化（全体に対して）
    if args.filter == "butter":
        qpos = mf.filter_qpos(qpos, fps, args.cut_root_xy, args.cut_root_z, args.cut_rot, args.cut_joint)
    elif args.filter == "ma":
        qpos = mf.moving_average_qpos(qpos, args.smooth)
    report["filter"] = args.filter if args.filter != "ma" else f"ma{args.smooth}"

    # 4. 区間の切り出し
    s = int(round(args.start * fps))
    e = len(qpos) if args.end is None else min(len(qpos), int(round(args.end * fps)))
    qpos = qpos[s:e].copy()
    print(f"[cut] {s}〜{e} フレーム → {len(qpos)} フレーム ({len(qpos) / fps:.2f} 秒)")

    # 5. 接地補正と足滑り
    model = load_model(args.robot_xml or default_robot_xml())
    assert model.nq == qpos.shape[1], f"nq={model.nq} と qpos 列数 {qpos.shape[1]} が不一致"
    dist, fpos = foot_state(model, qpos)
    fv = mf.foot_speed_xy(fpos, fps)
    print(f"[ground] 補正前 足-床距離: min {dist.min():+.3f} / median {np.median(dist.min(1)):+.3f} [m]")
    if args.ground != "none":
        dz, contact, ginfo = mf.ground_offsets(dist, fv, fps, args.contact_h, args.contact_v, mode=args.ground)
    else:
        dz, contact, ginfo = np.zeros(len(qpos)), (dist < args.contact_h) & (fv < args.contact_v), {"mode": "none"}
    qpos[:, 2] += dz + args.z_offset
    dist, fpos = foot_state(model, qpos)
    fv = mf.foot_speed_xy(fpos, fps)
    contact = (dist < args.contact_h) & (fv < args.contact_v)
    slide = mf.foot_slide(fv, contact)
    report["ground"] = ginfo
    report["after_ground"] = {
        "min_dist_m": float(dist.min()),
        "lowest_foot_median_m": float(np.median(dist.min(1))),
        "penetration_gt3cm_frames": int((dist.min(1) < -0.03).sum()),
        "float_gt5cm_while_both_feet_still_frames": int(((dist.min(1) > 0.05) & (fv.max(1) < 0.1)).sum()),
        "contact_ratio": float(contact.any(1).mean()),
        "foot_slide_mean_mps": float(slide[contact.any(1)].mean()) if contact.any() else 0.0,
    }
    flags = {"slide_gt_0.2mps": slide > 0.2, "penetration_gt_3cm": dist.min(1) < -0.03}
    report["per_second"] = mf.per_second(flags, fps, t0=args.start)
    print(f"[ground] 補正 {ginfo} → 補正後 min {dist.min():+.3f} m, 接地中の足滑り 平均 "
          f"{report['after_ground']['foot_slide_mean_mps']:.3f} m/s")

    # 6. 保存
    csv = mf.csv_from_qpos(qpos).astype(np.float32)
    csv_path = out_dir / f"{args.name}.csv"
    np.savetxt(csv_path, csv, delimiter=",", fmt="%.6f")
    with open(out_dir / f"{args.name}.pkl", "wb") as fp:
        pickle.dump({"fps": fps, "root_pos": csv[:, 0:3], "root_rot": csv[:, 3:7], "dof_pos": csv[:, 7:],
                     "local_body_pos": None, "link_body_list": None}, fp)
    (out_dir / f"{args.name}_prep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[save] {csv_path}  ({csv.shape[0]} 行 x {csv.shape[1]} 列, {fps:g} fps)")
    print(f"[save] {out_dir / (args.name + '_prep_report.json')}")
    bad_secs = [r["sec"] for r in report["per_second"]]
    if bad_secs:
        print(f"[warn] 足滑り・めり込みが目立つ秒: {bad_secs[:30]}")

    if not args.no_video:
        vid = out_dir / f"{args.name}_retarget_preview.mp4"
        render_preview(model, qpos, fps, str(vid), args.width, args.height,
                       args.cam_azimuth, args.cam_elevation, args.cam_distance)
        print(f"[video] {vid}")


if __name__ == "__main__":
    main()
