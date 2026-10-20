#!/usr/bin/env python3
"""シミュレーション専用の学習ラッパー。【新規】

unitree_rl_mjlab の scripts/train.py をそのまま使い、学習を始める前に設定を少しだけ変える。
既定値は実機に持っていく前提（sim-to-real）で、外乱や摩擦のばらつきが大きめに設定されている。
今回は物理シミュレーションの中で踊れれば十分なので、学習を楽にするために次を緩める。

  --friction-min / --friction-max : 足裏の摩擦のばらつき（既定 0.3〜1.2 → 0.6〜1.0）
  --push-scale                    : 体を押す外乱の強さの倍率（既定 1.0 → 0.5。0 で外乱なし）
  --term-threshold                : 「転倒」とみなす高さのずれ [m]（既定 0.25。お手本が粗い間だけ 0.35 などに緩める）

【2026-10-18 追加】動きの「キレ」と追従の細かさを調整する（指定しなければリポジトリの既定値のまま）
  --action-rate-weight            : 動作の変化の大きさにかかる罰則の重み（既定 -0.1）。
                                    0 に近づけるほど動きが鋭くなるが、小刻みな震えが出やすい
  --body-pos-std                  : 体の位置の報酬の許容幅 [m]（既定 0.3）。小さいほど細かいずれまで直そうとする
  --body-ori-std                  : 体の向きの報酬の許容幅 [rad]（既定 0.4）
  --root-pos-std                  : 立ち位置（根の位置・ワールド座標）の報酬の許容幅 [m]（既定 0.3）。締めるとドリフトが減る
  ※ 許容幅を締めるのは、すでに上手に踊れる方策の追加学習（Resume）で使う。最初から締めると学習が進みにくい

それ以外の引数は train.py と同じ（--motion-file, --env.scene.num-envs, --agent.max-iterations など）。
本番の評価（render_policy.py）は、摩擦を固定値にして行う。

使い方（unitree_rl_mjlab のフォルダで実行）:
  python $P/scripts/train_sim.py Unitree-G1-Tracking --motion-file src/assets/motions/g1/x.npz \
      --env.scene.num-envs 4096 --agent.max-iterations 30000 --push-scale 0.5
"""

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())                          # src.tasks を読めるように
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))  # train.py を読めるように


def relax(env_cfg, a):
    ev = env_cfg.events
    if "foot_friction" in ev:
        ev["foot_friction"].params["ranges"] = (a.friction_min, a.friction_max)
    if "push_robot" in ev:
        if a.push_scale <= 0:
            ev.pop("push_robot")
        else:
            vr = ev["push_robot"].params["velocity_range"]
            ev["push_robot"].params["velocity_range"] = {
                k: (lo * a.push_scale, hi * a.push_scale) for k, (lo, hi) in vr.items()}
    if a.term_threshold is not None:
        for k in ("anchor_pos", "ee_body_pos"):
            if k in env_cfg.terminations:
                env_cfg.terminations[k].params["threshold"] = a.term_threshold
    rw = env_cfg.rewards
    if a.action_rate_weight is not None:
        rw["action_rate_l2"].weight = a.action_rate_weight
    if a.body_pos_std is not None:
        rw["motion_body_pos"].params["std"] = a.body_pos_std
    if a.body_ori_std is not None:
        rw["motion_body_ori"].params["std"] = a.body_ori_std
    if a.root_pos_std is not None:
        rw["motion_global_root_pos"].params["std"] = a.root_pos_std


def describe(env_cfg):
    ev, tm = env_cfg.events, env_cfg.terminations
    fr = ev["foot_friction"].params["ranges"] if "foot_friction" in ev else None
    push = ev["push_robot"].params["velocity_range"]["x"] if "push_robot" in ev else "なし"
    th = {k: tm[k].params.get("threshold") for k in ("anchor_pos", "ee_body_pos") if k in tm}
    rw = env_cfg.rewards
    rew = (f"action_rate_l2 重み {rw['action_rate_l2'].weight} / "
           f"motion_body_pos std {rw['motion_body_pos'].params.get('std')} / "
           f"motion_body_ori std {rw['motion_body_ori'].params.get('std')} / "
           f"motion_global_root_pos std {rw['motion_global_root_pos'].params.get('std')}")
    return f"足裏摩擦 {fr} / 外乱(x方向の速度) {push} / 終了しきい値 {th} / 報酬: {rew}"


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--friction-min", "--friction_min", dest="friction_min", type=float, default=0.6)
    pre.add_argument("--friction-max", "--friction_max", dest="friction_max", type=float, default=1.0)
    pre.add_argument("--push-scale", "--push_scale", dest="push_scale", type=float, default=0.5)
    pre.add_argument("--term-threshold", "--term_threshold", dest="term_threshold", type=float, default=None)
    pre.add_argument("--action-rate-weight", "--action_rate_weight", dest="action_rate_weight", type=float, default=None)
    pre.add_argument("--body-pos-std", "--body_pos_std", dest="body_pos_std", type=float, default=None)
    pre.add_argument("--body-ori-std", "--body_ori_std", dest="body_ori_std", type=float, default=None)
    pre.add_argument("--root-pos-std", "--root_pos_std", dest="root_pos_std", type=float, default=None)
    a, rest = pre.parse_known_args()

    import mjlab
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import tyro
    from mjlab.tasks.registry import list_tasks

    import train as T  # unitree_rl_mjlab/scripts/train.py

    chosen_task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        args=rest, add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS,
    )
    default = T.TrainConfig.from_task(chosen_task)
    relax(default.env, a)  # コマンドラインの --env.* 指定はこの後に上書きされる
    args = tyro.cli(T.TrainConfig, args=remaining, default=default,
                    prog=sys.argv[0] + f" {chosen_task}", config=mjlab.TYRO_FLAGS)
    print(f"[train_sim] タスク {chosen_task}")
    print(f"[train_sim] {describe(args.env)}", flush=True)
    T.launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
