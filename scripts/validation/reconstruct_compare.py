"""
reconstruct_compare.py — Stage 1 重建对比动画 (离屏 gif, 无需介面)
==================================================================
一段轨迹, 每帧叠放两个 FR3:
  - 原始 q   (蓝色半透明) = ground truth
  - 重建 q'  (红色半透明) = latent -> decoder 还原
两臂重合 = 重建准; 分开 = 误差来源, 一眼看出偏在哪。

离屏算繪 (server 无显示器也能跑), 相机绕转, 输出 gif。

用法: python scripts/reconstruct_compare.py
输出: validation_out/reconstruct_compare.gif
若离屏失败: MUJOCO_GL=egl python scripts/reconstruct_compare.py
"""
import os, sys
import numpy as np
import torch
import mujoco

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
sys.path.insert(0, PROJECT_ROOT)
from fk_encoder.fk_encoder import FKEncoder
from layer2.embedding import MLPEmbedding
from layer2.decoder import Decoder

MENAGERIE = os.path.expanduser("~/mujoco_menagerie")
FR3_XML = os.path.join(MENAGERIE, "franka_fr3/fr3.xml")
ROBOTS_YAML = os.path.join(PROJECT_ROOT, "configs/robots.yaml")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints/stage1_franka.pt")
OUT_DIR = os.path.join(PROJECT_ROOT, "validation_out")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARM_DOF = 7; N_TRAJ = 60
W_img, H_img = 640, 480
DEMO_POSE = np.array([0.0,-0.5,0.0,-1.8,0.0,1.2,0.0], dtype=np.float32)


def build_two_fr3():
    """一个场景放两份 FR3 (原始/重建), 分别改颜色。回传 (model, qadr_a, qadr_b)。"""
    spec = mujoco.MjSpec.from_file(FR3_XML)
    # attach 第二份 FR3 到 worldbody (前缀 _rec)
    spec2 = mujoco.MjSpec.from_file(FR3_XML)
    frame = spec.worldbody.add_frame()
    frame.attach_body(spec2.body("base"), "", "_rec")

    # 加桌面 + 背景
    try:
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.7,0.8,0.9], rgb2=[0.95,0.97,1.0], width=256, height=256)
    except Exception:
        pass
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6,0.6,0.02],
                            pos=[0.2,0,-0.02], rgba=[0.82,0.71,0.55,1])

    model = spec.compile()

    # 诊断: 印出所有 body 名 (用来确认 _rec 后缀 + 上色是否正确)
    print("[diag] body names:", [model.body(i).name for i in range(model.nbody)])

    # 给两份机器人的 geom 上色: 原始蓝, 重建(_rec)红, 半透明
    n_blue = n_red = 0
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = model.body(bid).name
        if "_rec" in bname:
            model.geom_rgba[gid] = [0.85, 0.2, 0.2, 0.55]   # 红半透明
            n_red += 1
        elif bname.startswith("fr3_") or bname == "base":
            model.geom_rgba[gid] = [0.2, 0.4, 0.85, 0.55]   # 蓝半透明
            n_blue += 1
    print(f"[diag] colored {n_blue} blue (original), {n_red} red (reconstructed) geoms")

    # 找两组手臂关节的 qpos 地址
    qadr_a, qadr_b = [], []
    for i in range(model.njnt):
        jn = model.joint(i).name
        if jn and jn.startswith("fr3_joint") and "_rec" not in jn:
            qadr_a.append((jn, model.jnt_qposadr[i]))
        elif jn and "fr3_joint" in jn and "_rec" in jn:
            qadr_b.append((jn, model.jnt_qposadr[i]))
    qadr_a = [a for _,a in sorted(qadr_a)]
    qadr_b = [a for _,a in sorted(qadr_b)]
    print(f"[diag] original arm joints: {len(qadr_a)}, reconstructed arm joints: {len(qadr_b)}")
    if len(qadr_a) != ARM_DOF or len(qadr_b) != ARM_DOF:
        print(f"  [WARN] 关节数不对 (期望各 {ARM_DOF})! 贴 body names 给我修。")
    return model, qadr_a, qadr_b


def main():
    # 载入 encoder + 训练好的 W/decoder
    enc = FKEncoder("fr3", ROBOTS_YAML, project_root=PROJECT_ROOT, device=DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE)
    LD = ck["latent_dim"]
    W = MLPEmbedding(18, LD).to(DEVICE); W.load_state_dict(ck["W"]); W.eval()   # 输入 18
    # decoder: depth 从 checkpoint 推不出, 用 train 的设定 (hidden256 depth2 use_q_prev)
    dec = Decoder(LD, m_dim=3, out_dim=ARM_DOF, hidden=256, depth=2, use_q_prev=True).to(DEVICE)
    dec.load_state_dict(ck["decoder"]); dec.eval()
    m = enc.m.to(DEVICE).unsqueeze(0)
    print(f"[init] latent_dim={LD}")

    # 轨迹 + 重建
    t = np.linspace(0, 2*np.pi, N_TRAJ)
    traj = np.stack([DEMO_POSE[k] + 0.4*np.sin(t+k) for k in range(ARM_DOF)], axis=1).astype(np.float32)
    with torch.no_grad():
        Q = torch.from_numpy(traj).to(DEVICE)
        fk = enc.forward(Q); z = W(fk[:, :18])
        Qp = dec(z, m.expand(N_TRAJ,-1), q_prev=Q).cpu().numpy()   # 重建 q'
    print("[data] trajectory reconstructed")

    model, qadr_a, qadr_b = build_two_fr3()
    data = mujoco.MjData(model)

    try:
        renderer = mujoco.Renderer(model, height=H_img, width=W_img)
    except Exception as e:
        print(f"[ERROR] 离屏 renderer 失败: {e}")
        print("  试: MUJOCO_GL=egl python scripts/reconstruct_compare.py")
        return

    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance = 2.4; cam.elevation = -18; cam.lookat[:] = [0.1,0,0.5]

    frames = []
    for i in range(N_TRAJ):
        data.qpos[:] = 0.0
        for k,adr in enumerate(qadr_a): data.qpos[adr] = traj[i,k]   # 原始
        for k,adr in enumerate(qadr_b): data.qpos[adr] = Qp[i,k]     # 重建
        mujoco.mj_forward(model, data)
        cam.azimuth = 130 + i*(360.0/N_TRAJ)   # 相机绕转一圈
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    _save_gif(frames, os.path.join(OUT_DIR, "reconstruct_compare.gif"), fps=15)
    print(f"[done] reconstruct_compare.gif  (蓝=原始, 红=重建, 重合=准)")


def _save_gif(frames, path, fps=15):
    try:
        from PIL import Image
        imgs=[Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000/fps), loop=0)
    except ImportError:
        import imageio; imageio.mimsave(path, frames, fps=fps)


if __name__ == "__main__":
    main()