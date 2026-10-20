#!/usr/bin/env python3
"""保存済みの GVHMR の推定結果 (hmr4d_results.pt) から、人のメッシュ動画をカメラの向きだけ変えて描き直す。【2026-10-18 新規】

GVHMR の tools/demo/demo.py の render_global() と同じ処理（同じメッシュ・床・照明・24mm レンズ）で描画し、
カメラの置き場所だけを変える。推定（重い処理）はやり直さない。

元の 2_global.mp4 のカメラは「最初のフレームの体の向き」から 45° の位置に固定されている。
この動画は最初と最後で体の向きが平均から大きくずれているため、結果として背中側から撮った画になっていた。
このスクリプトは全フレームの体の向きを平均し、その正面（--view_deg 0）にカメラを置く。

使い方（GVHMR のフォルダで、GVHMR の venv の python で実行）:
  cd /content/GVHMR
  /content/venv_gvhmr/bin/python $P/scripts/render_gvhmr_front.py \
      --pred $P/outputs/gvhmr/the_feels_full/hmr4d_results.pt \
      --video $P/input/the_feels_full.mp4 --out $P/outputs/gvhmr/the_feels_full/2_global_front.mp4
  --view_deg : 正面からの水平の回り込み角 [deg]（0 = 真正面、180 = 真後ろ、45 = 斜め前）
  --max_frames 90 : 試しに最初の 3 秒だけ描く
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from einops import einsum

sys.path.insert(0, str(Path.cwd()))  # GVHMR のフォルダから実行する前提
from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import get_video_lwh, get_writer
from hmr4d.utils.vis.renderer import Renderer, get_ground_params_from_points
from pytorch3d.renderer import PointLights
from pytorch3d.renderer.cameras import look_at_rotation
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="hmr4d_results.pt")
    p.add_argument("--video", required=True, help="元動画（30fps に変換したもの。解像度とフレーム数をここから取る）")
    p.add_argument("--out", required=True, help="出力 mp4")
    p.add_argument("--view_deg", type=float, default=0.0, help="正面からの回り込み角 [deg]（0=正面）")
    p.add_argument("--beta", type=float, default=2.0, help="カメラ距離の倍率（demo.py と同じ 2.0）")
    p.add_argument("--cam_height_degree", type=float, default=20.0, help="見下ろす角度 [deg]（demo.py と同じ 20）")
    p.add_argument("--max_frames", type=int, default=None, help="試し描画用（先頭から何フレーム描くか）")
    return p.parse_args()


def facing_xz(joints):
    """各フレームの体の正面方向（xz 平面の単位ベクトル）。GVHMR の compute_T_ayfz2ay と同じ定義（腰と肩の左右から）"""
    rl = (joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]) + (joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]])  # 体の左向き
    fwd = torch.stack([-rl[:, 1], rl[:, 0]], dim=-1)  # 左 × 上 = 正面
    return fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def main():
    a = parse_args()
    pred = torch.load(a.pred)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    # demo.py の render_global と同じ前処理（最初のフレームの位置を原点・床に合わせ、体の向きを +Z に）
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
    offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]
    offset[1] = verts[:, :, [1]].min()
    verts = verts - offset
    T = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
    verts = apply_T_on_points(verts, T)
    joints = einsum(J_regressor, verts, "j v, l v i -> l j i")

    # 全フレームの体の向きの平均 → その正面（から view_deg 回り込んだ位置）にカメラを置く
    f = facing_xz(joints).mean(0)
    f = f / f.norm()
    th = np.deg2rad(a.view_deg)
    c, s = float(np.cos(th)), float(np.sin(th))
    vx, vz = c * f[0].item() + s * f[1].item(), -s * f[0].item() + c * f[1].item()
    fr = facing_xz(joints)
    dev = torch.rad2deg(torch.acos((fr @ f).clamp(-1, 1)))
    print(f"[info] 平均の向き (x, z) = ({f[0]:+.3f}, {f[1]:+.3f}) / 正面±45° に入るフレーム {(dev < 45).float().mean():.0%}")

    # demo.py の get_global_cameras_static と同じ計算（カメラの方向ベクトルだけ差し替え）
    targets = verts.mean(1).cpu()
    targets[:, 1] = 0
    center = targets.mean(0)
    scale = max(torch.norm(targets - center, dim=-1).max().item(), 1.0) * a.beta
    vec = torch.tensor([vx, 0.0, vz]).float()
    pos = center + vec * scale
    pos[1] = scale * np.tan(np.deg2rad(a.cam_height_degree)) + 1.0
    L = verts.shape[0]
    positions = pos.unsqueeze(0).repeat(L, 1)
    centers = center.unsqueeze(0).repeat(L, 1)
    centers[:, 1] = 1.0
    R = look_at_rotation(positions, centers).mT
    Tt = -(R @ positions.unsqueeze(-1)).squeeze(-1)
    lights = PointLights(device="cuda", location=[pos.tolist()])

    length, width, height = get_video_lwh(a.video)
    _, _, K = create_camera_sensor(width, height, 24)
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    sc, cx, cz = get_ground_params_from_points(joints[:, 0], verts)
    renderer.set_ground(sc * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    n = min(length, L) if a.max_frames is None else min(a.max_frames, length, L)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = get_writer(a.out, fps=30, crf=23)
    for i in tqdm(range(n), desc="Rendering (front)"):
        cams = renderer.create_camera(R[i], Tt[i])
        writer.write_frame(renderer.render_with_ground(verts[[i]], color[None], cams, lights))
    writer.close()
    print(f"[save] {a.out}  ({n} フレーム, {width}x{height})")


if __name__ == "__main__":
    main()
