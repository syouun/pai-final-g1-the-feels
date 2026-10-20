#!/usr/bin/env python3
"""学習済みの方策で G1 を物理シミュレーション上で踊らせ、動画と評価値を保存する。【改訂版】

unitree_rl_mjlab 付属の play.py はビューアを起動し続けるため、Colab では使いにくい。
このスクリプトは表示なしでダンスを先頭から 1 回だけ実行する。

旧版からの主な修正:
  - 評価の条件を固定: play モードでも残っていた「足裏摩擦・重心位置・関節センサーのずれ」の
    ランダム化を外し、摩擦は --friction の値に固定する（毎回同じ結果になる＝再現できる）
  - 転倒したら、そこで記録を止める（旧版はリセット後にお手本が冒頭へ巻き戻ったまま録画を続けていた）
  - 体の位置の誤差を、学習の報酬と同じ「アンカー（胴体）基準」で測る（旧版はワールド座標で、
    根の位置のドリフトまで誤差に入っていた）。ドリフトは別の項目として記録する

出力:
  - <out>.mp4 : 動画（--no-video で省略）
  - <out>.json: 評価値（成功したか、転倒した時刻、誤差の平均など）
  - <out>.csv : フレームごとの誤差（レポートのグラフ用）

使い方（unitree_rl_mjlab のフォルダで実行）:
  python render_policy.py --task Unitree-G1-Tracking \
      --motion-file src/assets/motions/g1/the_feels_20s.npz \
      --checkpoint logs/rsl_rl/g1_tracking/<日時>/model_5000.pt \
      --out outputs/policy_5000.mp4 --friction 0.8
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.getcwd())  # unitree_rl_mjlab の src パッケージを読めるようにする

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Unitree-G1-Tracking", help="学習に使ったタスク名と同じにする")
    p.add_argument("--motion-file", required=True, help="学習に使った npz")
    p.add_argument("--checkpoint", required=True, help="model_XXXX.pt")
    p.add_argument("--out", required=True, help="出力 mp4 のパス（.json / .csv も同じ名前で出る）")
    p.add_argument("--friction", type=float, default=0.8, help="足裏の摩擦係数（固定値）")
    p.add_argument("--no-video", action="store_true", help="動画を書き出さない（評価値だけ欲しいとき。速い）")
    p.add_argument("--keep-going", action="store_true",
                   help="転倒しても止めない（リセット後はお手本が冒頭に戻るので、動画は提出には使わないこと）")
    p.add_argument("--no-terminations", action="store_true",
                   help="転倒判定そのものを切る（転んだあとの様子をそのまま撮りたいとき）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=None, help="カメラの水平角 [deg]")
    p.add_argument("--distance", type=float, default=None, help="カメラ距離 [m]")
    p.add_argument("--elevation", type=float, default=None, help="カメラの仰角 [deg]（負で見下ろす。既定 -5）")
    # 【2026-10-18 追加】診断用
    p.add_argument("--t-end", type=float, default=None,
                   help="この秒数で打ち切る（転倒の前後だけ短い動画を見たいとき。誤差の集計もここまで）")
    p.add_argument("--dump-bodies", action="store_true",
                   help="部位ごとの高さのずれ（お手本 − ロボット）と、足の水平速度を <out>_bodies.csv に書く")
    # 【2026-10-19 追加】提出用動画
    p.add_argument("--no-ghost", action="store_true",
                   help="お手本の半透明表示（緑のゴースト）を描かない。ロボット本体だけを映す（物理計算には影響しない）")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    import mjlab.tasks  # noqa: F401  タスク登録
    import src.tasks  # noqa: F401  unitree のタスク登録
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    # play=True: 外乱なし・観測ノイズなし・初期姿勢のランダム化なし・お手本の先頭から再生
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["motion"].motion_file = str(Path(args.motion_file).resolve())

    # 評価条件の固定（play モードでも起動時のランダム化は残っているため、ここで外す）
    ev = env_cfg.events
    for k in ("base_com", "encoder_bias", "push_robot"):
        ev.pop(k, None)
    if "foot_friction" in ev:
        ev["foot_friction"].params["ranges"] = (args.friction, args.friction)

    env_cfg.viewer.width, env_cfg.viewer.height = args.width, args.height
    if args.azimuth is not None:
        env_cfg.viewer.azimuth = args.azimuth
    if args.distance is not None:
        env_cfg.viewer.distance = args.distance
    if args.elevation is not None:
        env_cfg.viewer.elevation = args.elevation
    # 【2026-10-18 追加】ee_body_pos が見ている部位（手首・足首など）を、終了条件を外す前に控えておく
    ee_names = None
    try:
        ee_names = list(env_cfg.terminations["ee_body_pos"].params.get("body_names") or [])
    except Exception:
        pass
    if args.no_terminations:
        env_cfg.terminations = {}
    if args.no_ghost:
        # mjlab のモーションコマンドのデバッグ表示（viz.mode="ghost"）を切る。描画だけの設定
        env_cfg.commands["motion"].debug_vis = False

    render_mode = None if args.no_video else "rgb_array"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    cmd = env.unwrapped.command_manager.get_term("motion")
    total = int(cmd.motion.time_step_total)
    fps = 1.0 / env.unwrapped.step_dt
    print(f"[info] モーション {total} ステップ ({total / fps:.1f} 秒, {fps:.0f} fps) / 摩擦 {args.friction}")

    env.reset()
    # 【2026-10-17 修正】リセット直後は、お手本の体の位置（胴体基準に直したもの = body_pos_relative_w）が
    # まだ計算されておらず 0 のまま（計算されるのは env.step の最後）。このまま 1 ステップ目の終了判定に入ると、
    # 手首の高さのずれ（約 0.8 m）を「転倒」と誤判定し ee_body_pos で 0.02 秒で止まる。
    # time_steps を 1 つ戻してから更新関数を呼び、お手本の先頭フレームのまま値だけ計算しておく。
    cmd.time_steps -= 1
    cmd._update_command()
    # 開始時点のロボットとお手本のずれ（リセットでお手本の姿勢に置かれているかの確認）
    d0 = (cmd.anchor_pos_w - cmd.robot_anchor_pos_w)[0]
    print(f"[info] 開始時のずれ（お手本 − ロボット, 胴体）: x {d0[0]:+.3f} / y {d0[1]:+.3f} / z {d0[2]:+.3f} m")
    if args.no_terminations:
        print("[info] 終了条件を外して実行します（転倒しても止めない。success は判定しない）")
    tm = env.unwrapped.termination_manager

    def fired_terms():
        """いま True になっている終了条件の名前（どの条件で止まったかを記録する）"""
        names = []
        for n in getattr(tm, "active_terms", []):
            try:
                if bool(tm.get_term(n)[0].item()):
                    names.append(n)
            except Exception:
                pass
        return names

    # 【2026-10-18 追加】部位の名前（コマンドが追跡している体の並び）
    body_names = list(getattr(cmd.cfg, "body_names", []) or [])
    def body_z_err():
        """部位ごとの高さのずれ（お手本 − ロボット）[m]。ee_body_pos の判定と同じ量"""
        return (cmd.body_pos_relative_w[0, :, 2] - cmd.robot_body_pos_w[0, :, 2]).cpu().numpy()
    feet_idx = [i for i, n in enumerate(body_names) if "ankle" in n.lower()]
    brows, prev_feet = [], None
    fall_bodies = None
    z_prev = None   # 【2026-10-19 修正】1 ステップ前の高さのずれ。転倒した step の中で環境は自動でリセットされるので、
                    # step の後に読むとリセット後の値になる。転倒の 0.02 秒前の値で代用する

    frames, rows = [], []
    arates, prev_act = [], None   # 【2026-10-18 追加】動作の変化量（震え・キレの目安）
    fall_reason = None
    falls, first_fall = 0, None
    with torch.no_grad():
        obs = env.get_observations()
        for t in range(total - 1):
            act = policy(obs)
            if prev_act is not None:
                arates.append((act - prev_act).abs().mean().item())
            prev_act = act.clone()
            try:
                z_prev = body_z_err()
            except Exception:
                z_prev = None
            obs, _, dones, _ = env.step(act)
            fell = bool(dones[0].item())
            if fell:
                falls += 1
                if first_fall is None:
                    first_fall = (t + 1) / fps
                    fall_reason = fired_terms()
                    print(f"[fall] {first_fall:.2f} 秒で終了条件に該当: {fall_reason or '（名前を取得できず）'}")
                    # 【2026-10-18 追加・10-19 修正】どの部位の高さがずれていたか（転倒の 0.02 秒前、大きい順に 4 つ）
                    try:
                        z = z_prev
                        order = np.argsort(-np.abs(z))[:4]
                        fall_bodies = {(body_names[i] if i < len(body_names) else str(i)): round(float(z[i]), 3) for i in order}
                        print(f"[fall] 転倒の 0.02 秒前に高さのずれが大きかった部位（お手本 − ロボット [m]）: {fall_bodies}")
                    except Exception as e:  # 診断なので、失敗しても評価は続ける
                        print(f"[fall] 部位の取得に失敗: {e}")
                if not args.keep_going:
                    break
            if render_mode:
                frames.append(env.unwrapped.render())
            jerr = (cmd.joint_pos - cmd.robot_joint_pos).abs().mean().item()
            perr = (cmd.body_pos_relative_w - cmd.robot_body_pos_w).norm(dim=-1).mean().item()
            drift = (cmd.anchor_pos_w - cmd.robot_anchor_pos_w)[:, :2].norm(dim=-1).item()
            rows.append(((t + 1) / fps, jerr, perr, drift))
            if args.dump_bodies:
                z = body_z_err()
                fp = cmd.robot_body_pos_w[0, feet_idx, :2].cpu().numpy() if feet_idx else np.zeros((0, 2))
                fv = (np.linalg.norm(fp - prev_feet, axis=-1) * fps) if prev_feet is not None else np.zeros(len(feet_idx))
                prev_feet = fp
                brows.append(np.concatenate([[(t + 1) / fps], z, fv]))
            if args.t_end is not None and (t + 1) / fps >= args.t_end:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if render_mode and frames:
        import mediapy
        mediapy.write_video(str(out), frames, fps=fps)

    if args.dump_bodies and brows:
        hdr = ["time_s"] + [f"zerr_{n}" for n in body_names] + [f"robot_xyspeed_{body_names[i]}" for i in feet_idx]
        np.savetxt(out.with_name(out.stem + "_bodies.csv"), np.array(brows), delimiter=",",
                   header=",".join(hdr), comments="", fmt="%.4f")

    arr = np.array(rows) if rows else np.zeros((1, 4))
    np.savetxt(out.with_suffix(".csv"), arr, delimiter=",",
               header="time_s,joint_err_rad,body_pos_err_rel_m,root_xy_drift_m", comments="",
               fmt=["%.3f", "%.5f", "%.5f", "%.5f"])
    duration = round(total / fps, 2)
    summary = {
        "checkpoint": str(args.checkpoint),
        "motion_file": str(args.motion_file),
        "task": args.task,
        "friction": args.friction,
        "duration_s": duration,
        "terminations_disabled": bool(args.no_terminations),
        "success": None if args.no_terminations else first_fall is None,
        "fall_reason": fall_reason,
        "fall_bodies": fall_bodies,
        "ee_body_names": ee_names,
        "t_end": args.t_end,
        "first_fall_s": first_fall,
        "completed_ratio": round((first_fall or duration) / duration, 3),
        "falls": falls,
        "mean_joint_err_rad": round(float(arr[:, 1].mean()), 4),
        "mean_body_pos_err_rel_m": round(float(arr[:, 2].mean()), 4),
        "max_body_pos_err_rel_m": round(float(arr[:, 2].max()), 4),
        "final_root_xy_drift_m": round(float(arr[-1, 3]), 3),
        "max_root_xy_drift_m": round(float(arr[:, 3].max()), 3),
        # 1 ステップ（0.02 秒）ごとの方策の出力（関節目標へのオフセット、スケール前）の変化の平均。大きいほど動きが鋭い／震えやすい
        "mean_action_rate": round(float(np.mean(arates)), 4) if arates else None,
        "p95_action_rate": round(float(np.percentile(arates, 95)), 4) if arates else None,
        "video": str(out) if render_mode else None,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    env.close()


if __name__ == "__main__":
    main()
