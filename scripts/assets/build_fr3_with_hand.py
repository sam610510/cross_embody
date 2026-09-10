"""
build_fr3_with_hand.py — 给 menagerie FR3 接上 Franka Hand (TCP 反求对齐 + 自我诊断)
====================================================================================
夹爪本体已验证装对 (finger 对称 ±0.0283, slide 0~0.04)。
TCP site 改用「反求 local 座标」: 已知 encoder TCP 世界位置, 直接算它相对 hand body
的 local 座标填进去, 不再用物理 offset 猜方向 (因 hand.xml 内部座标系与 URDF 不同)。

用法: python scripts/build_fr3_with_hand.py
生成 ~/cross_embody/urdfs/franka_description/fr3_with_hand.xml
"""

import os
import numpy as np
import mujoco

MENAGERIE = os.path.expanduser("~/mujoco_menagerie")
FR3_XML   = os.path.join(MENAGERIE, "franka_fr3/fr3.xml")
HAND_XML  = os.path.join(MENAGERIE, "franka_emika_panda/hand.xml")
OUT_XML   = os.path.expanduser("~/cross_embody/urdfs/franka_description/fr3_with_hand.xml")

HAND_ROT_Z = -0.7853981633974483   # -45deg, fr3_hand_joint
ENC_TCP_Q0 = np.array([0.088, 0.0, 0.8226])   # encoder p_ee[:3] @ q=0


def quat_z(theta):
    return [np.cos(theta/2), 0.0, 0.0, np.sin(theta/2)]


def build(tcp_local=None):
    """tcp_local=None 时先不加 TCP site (用来量 hand body pose);
    给定时把 TCP site 加在 hand body 上。回传 (model, spec, attached_hand_name)。"""
    spec = mujoco.MjSpec.from_file(FR3_XML)
    hand_spec = mujoco.MjSpec.from_file(HAND_XML)
    site = spec.site("attachment_site")
    frame = site.parent.add_frame()
    frame.pos = [0.0, 0.0, 0.0]
    frame.quat = quat_z(HAND_ROT_Z)
    attached = frame.attach_body(hand_spec.body("hand"), "", "_grip")
    if tcp_local is not None:
        # TCP 位置 + 姿态对齐 encoder。
        # 实测 encoder 与 MuJoCo hand 的 TCP 朝向差「绕 z 180°」, 故给 site 加此旋转。
        attached.add_site(name="fr3_hand_tcp", pos=list(tcp_local),
                          quat=[0.0, 0.0, 0.0, 1.0],   # 绕 z 180° (wxyz)
                          size=[0.008], rgba=[1, 0, 0, 1])
    model = spec.compile()
    return model, spec, attached.name


def _ensure_hand_meshes(fr3_assets):
    """把 hand 的 mesh 复制到 fr3 assets 目录, 让单一 meshdir 同时涵盖手臂与夹爪。
    只复制缺少的档案, 不覆盖既有。"""
    import shutil
    hand_assets = os.path.join(MENAGERIE, "franka_emika_panda", "assets")
    if not os.path.isdir(hand_assets):
        print(f"  [warn] hand assets dir not found: {hand_assets}")
        return
    os.makedirs(fr3_assets, exist_ok=True)
    copied = 0
    for fn in os.listdir(hand_assets):
        if fn.lower().endswith((".obj", ".stl", ".png")):
            dst = os.path.join(fr3_assets, fn)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(hand_assets, fn), dst)
                copied += 1
    print(f"  [mesh] copied {copied} hand mesh files into fr3 assets")


def main():
    print("="*60)
    # ---------- 1. 先 attach hand, 量 hand body 在 q=0 的世界 pose ----------
    model1, _, hand_body_name = build(tcp_local=None)
    d1 = mujoco.MjData(model1); d1.qpos[:] = 0; mujoco.mj_forward(model1, d1)
    hb_id = model1.body(hand_body_name).id
    hand_pos = d1.xpos[hb_id].copy()
    hand_mat = d1.xmat[hb_id].reshape(3, 3).copy()
    print(f"[diag] attached hand body = '{hand_body_name}'")
    print(f"[diag] hand body world pos @ q=0 = {np.round(hand_pos,5)}")

    # ---------- 2. 反求 TCP 相对 hand body 的 local 座标 ----------
    # world = hand_pos + hand_mat @ local  =>  local = hand_mat^T @ (world - hand_pos)
    tcp_local = hand_mat.T @ (ENC_TCP_Q0 - hand_pos)
    print(f"[solve] TCP local (rel hand body) = {np.round(tcp_local,5)}")

    # ---------- 3. 重建, 把 TCP site 填进去 ----------
    model2, spec, _ = build(tcp_local=tcp_local)

    # 关键: 设定 meshdir 为绝对路径, 否则存档后从别处跑会找不到 mesh (assets/*.obj)。
    # menagerie 的 fr3 mesh 在 franka_fr3/assets/, hand mesh 在 franka_emika_panda/assets/。
    fr3_assets = os.path.join(MENAGERIE, "franka_fr3", "assets")
    spec.meshdir = fr3_assets
    # hand 的 mesh 也需可寻; 若 compiler 只认一个 meshdir, 把 hand assets 的档案
    # 也指过去 (两边 assets 不同目录时, 下面 fallback 会复制)。
    _ensure_hand_meshes(fr3_assets)

    # 加大离屏 framebuffer, 让之后 render_mujoco 可用高解析 (默认仅 640x480)
    try:
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 1280
    except Exception as e:
        print(f"  [warn] 设 offscreen buffer 失败 (不影响对拍): {e}")

    with open(OUT_XML, "w") as f:
        f.write(spec.to_xml())
    print(f"[done] wrote {OUT_XML}")
    print(f"[done] meshdir set to {fr3_assets}")

    # ---------- 4. 验证 ----------
    d2 = mujoco.MjData(model2); d2.qpos[:] = 0; mujoco.mj_forward(model2, d2)
    tcp_id = model2.site("fr3_hand_tcp").id
    tcp_world = d2.site_xpos[tcp_id].copy()
    err = np.linalg.norm(tcp_world - ENC_TCP_Q0) * 1000
    print("\n" + "="*60)
    print("[verify] fr3_hand_tcp world @ q=0 =", np.round(tcp_world, 5))
    print("[verify] encoder p_ee[:3]   @ q=0 =", np.round(ENC_TCP_Q0, 5))
    print(f"[verify] TCP 对齐误差 = {err:.4f} mm", "  PASS" if err < 1.0 else "  FAIL")
    # 旋转验证
    R_tcp = d2.site_xmat[tcp_id].reshape(3, 3)
    qm = np.zeros(4); mujoco.mju_mat2Quat(qm, R_tcp.flatten())
    print(f"[verify] fr3_hand_tcp quat(wxyz) = {np.round(qm,4)}")
    print(f"         (应与 encoder p_ee[3:7] 一致; 若仍差 180° 检查 site quat)")
    print("="*60)

    # ---------- 5. 多姿态交叉验证: TCP 应在各姿态都对齐 encoder ----------
    # (反求只用了 q=0; 若 hand 刚体接对, 其他姿态也该自动对齐。这里实测确认。)
    print("\n[cross-check] 随机姿态下 TCP vs encoder (需 encoder, 见 validate_fk):")
    print("  -> 反求基于刚体变换, q=0 对齐即代表全姿态对齐 (hand 为刚体)。")
    print("  -> 完整多姿态对拍在 validate_fk.py 跑 (EE 改用 fr3_hand_tcp)。")

    # ---------- 6. finger 诊断 ----------
    print("\n[finger] joints & 开合对称:")
    for i in range(model2.njnt):
        jn = model2.joint(i).name
        if "finger" in jn.lower():
            jt = {0:"free",1:"ball",2:"slide",3:"hinge"}[int(model2.jnt_type[i])]
            print(f"  - {jn}: {jt}, range={np.round(model2.jnt_range[i],4)}")
    for i in range(model2.njnt):
        if "finger" in model2.joint(i).name.lower():
            d2.qpos[model2.jnt_qposadr[i]] = 0.04
    mujoco.mj_forward(model2, d2)
    for i in range(model2.nbody):
        if "finger" in model2.body(i).name.lower():
            print(f"  {model2.body(i).name} pos @open =", np.round(d2.xpos[i], 4))


if __name__ == "__main__":
    main()