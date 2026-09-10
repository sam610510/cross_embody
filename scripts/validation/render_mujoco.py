"""
render_mujoco.py — 用 MuJoCo 离屏算繪输出 FR3+夹爪 的静态图 + joint-0 掃描 gif
============================================================================
目的: 报告里放真实 MuJoCo 渲染 (带夹爪 FR3), 与 v_kp_test 的 keyvector 平滑性曲线
      逐帧对照, 证明你的 FK encoder 对应真实机器人几何。

gif 掃描设定与 v_kp_test.verify_smoothness_and_monotonicity 完全一致:
    BASE_POSE = [0, -0.5, 0, -1.8, 0, 1.2, 0]
    joint 0 从 -pi/2 扫到 +pi/2, 其他关节固定。
    每一帧精确对应 v_kp 曲线上的一个 x 座标。

离屏渲染 (EGL/osmesa), 不需开窗, server 无显示器也能跑。
若离屏失败: export MUJOCO_GL=egl

用法:
    conda activate cross_embody
    python scripts/render_mujoco.py

输出:
    validation_out/mj_q0.png              q=0 静态
    validation_out/mj_pose.png            BASE_POSE 静态
    validation_out/mj_joint0_sweep.gif    joint 0 从 -pi/2 到 +pi/2 掃描 (对照 v_kp)
"""

import os
import numpy as np
import mujoco

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
MJCF = os.path.join(PROJECT_ROOT, "urdfs/franka_description/fr3_with_hand.xml")
OUT_DIR = os.path.join(PROJECT_ROOT, "validation_out")
os.makedirs(OUT_DIR, exist_ok=True)

ARM_DOF = 7
W, H = 640, 480

# 基准姿态 (与 v_kp_test 的 BASE_POSE 一致)
BASE_POSE = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float64)

# ---- 掃描设定 (与 v_kp_test 对齐) ----
SWEEP_JOINT = 0            # 掃 joint 0
SWEEP_RANGE = np.pi / 2    # -pi/2 -> +pi/2
N_SWEEP = 120              # gif 帧数


def arm_qadr(model):
    named = []
    for i in range(model.njnt):
        jn = model.joint(i).name
        if jn and jn.startswith("fr3_joint"):
            named.append((jn, model.jnt_qposadr[i]))
    named.sort(key=lambda x: x[0])
    return [a for _, a in named[:ARM_DOF]]


def add_scene(spec):
    """给 model 加桌面 + 背景 + 补光, 让渲染好看 (不动对拍 model)。"""
    try:
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.7, 0.8, 0.9], rgb2=[0.95, 0.97, 1.0],
                         width=256, height=256)
    except Exception as e:
        print(f"  [warn] skybox 加失败 (不影响): {e}")

    try:
        spec.add_material(name="table_mat", rgba=[0.82, 0.71, 0.55, 1.0])
    except Exception:
        pass

    wb = spec.worldbody
    wb.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.6, 0.6, 0.02], pos=[0.2, 0.0, -0.02],
                material="table_mat", rgba=[0.82, 0.71, 0.55, 1.0])
    for dx, dy in [(-0.5, -0.5), (0.7, -0.5), (-0.5, 0.5), (0.7, 0.5)]:
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.35],
                    pos=[0.2 + dx * 0.55, dy * 0.9, -0.37], rgba=[0.6, 0.5, 0.38, 1.0])
    wb.add_light(pos=[0.3, 0.3, 1.5], dir=[-0.2, -0.2, -1],
                 diffuse=[0.5, 0.5, 0.5], specular=[0.2, 0.2, 0.2])


def set_pose(data, qadr, pose):
    data.qpos[:] = 0.0
    for k, adr in enumerate(qadr):
        data.qpos[adr] = pose[k]


def main():
    spec = mujoco.MjSpec.from_file(MJCF)
    add_scene(spec)
    try:
        model = spec.compile()
    except Exception as e:
        print(f"[warn] 加场景后 compile 失败, 回退到原 model: {e}")
        model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    qadr = arm_qadr(model)

    max_w = int(model.vis.global_.offwidth)
    max_h = int(model.vis.global_.offheight)
    w = min(W, max_w); h = min(H, max_h)
    if (w, h) != (W, H):
        print(f"[info] 渲染尺寸夹到 {w}x{h} (model 上限 {max_w}x{max_h})")

    try:
        renderer = mujoco.Renderer(model, height=h, width=w)
    except Exception as e:
        print(f"[ERROR] 离屏 renderer 建立失败: {e}")
        print("  试: MUJOCO_GL=egl python scripts/render_mujoco.py")
        return

    # 相机: 从上方斜看 base 旋转最清楚 (joint 0 是绕 z 的 base 转)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 2.2
    cam.elevation = -25      # 稍微俯视, 看清 base 绕 z 旋转
    cam.azimuth = 130
    cam.lookat[:] = [0.3, 0.0, 0.4]

    # ---- (1) q=0 静态 ----
    set_pose(data, qadr, np.zeros(ARM_DOF))
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    _save_png(renderer.render(), os.path.join(OUT_DIR, "mj_q0.png"))
    print("[done] mj_q0.png   (q=0 静态)")

    # ---- (2) BASE_POSE 静态 ----
    set_pose(data, qadr, BASE_POSE)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    _save_png(renderer.render(), os.path.join(OUT_DIR, "mj_pose.png"))
    print("[done] mj_pose.png (BASE_POSE 静态)")

    # ---- (3) joint-0 掃描 gif (对照 v_kp_test) ----
    # 与 v_kp_test 完全一致: linspace(-pi/2, +pi/2), 其他关节固定在 BASE_POSE
    angles = np.linspace(-SWEEP_RANGE, SWEEP_RANGE, N_SWEEP)
    frames = []
    for a in angles:
        set_pose(data, qadr, BASE_POSE)
        data.qpos[qadr[SWEEP_JOINT]] = a   # 只覆盖 joint 0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    _save_gif(frames, os.path.join(OUT_DIR, "mj_joint0_sweep.gif"), fps=20)
    print(f"[done] mj_joint0_sweep.gif ({N_SWEEP} 帧, joint 0: -pi/2 -> +pi/2)")

    print(f"\n[done] outputs in {OUT_DIR}")
    print("  报告并排: mj_joint0_sweep.gif <-> v_SE/v_SW/v_SEE_smoothness_labeled.png")
    print("  gif 每帧对应 v_kp 曲线的一个 joint-0 角度。")


def _save_png(img, path):
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(path, img)


def _save_gif(frames, path, fps=20):
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
    except ImportError:
        import imageio
        imageio.mimsave(path, frames, fps=fps)


if __name__ == "__main__":
    main()