"""
detect_keyjoints.py — 自动侦测 shoulder / elbow / wrist (基于旋转轴方向)
========================================================================
规则 (你的定义):
  - 基座轴 = z 轴朝上 [0,0,1]
  - shoulder: 沿链从 base 往末端, 第一个「轴不平行于 base 轴」的关节
  - elbow:    shoulder 之后, 第一个「轴平行于 shoulder 轴」的关节
  - wrist:    elbow 之后, 第一个「轴不平行于 elbow 轴」的关节
  - 平行判定: |axis_a · axis_b| > PARALLEL_THRESH (0.95)

只侦测 + 对照, 不改动任何现有设定。
用法: python scripts/detect_keyjoints.py
"""
import os, sys
import numpy as np
import torch
import pytorch_kinematics as pk

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
PARALLEL_THRESH = 0.95
BASE_AXIS = np.array([0.0, 0.0, 1.0])

# (名称, urdf 路径, 手动 key_links 供对照)
ROBOTS = [
    ("fr3",
     os.path.join(PROJECT_ROOT, "urdfs/franka_description/fr3.urdf"),
     {"shoulder": "fr3_link2", "elbow": "fr3_link4", "wrist": "fr3_link6"}),
    # 之后有 URDF 再加 so100 / ur5e
]


def is_parallel(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return abs(float(np.dot(a, b))) > PARALLEL_THRESH


def ordered_joints(chain):
    """回传从 base 到末端有序的 (joint_name, axis_world, child_link_name), 只含可动关节。
    关键: URDF 的 joint.axis 是 local frame 的, 各关节 local z 都可能相同 (如 FR3 全是 z),
    必须先做 FK (q=0), 把 local 轴用该 frame 的世界旋转转到世界座标, 才能正确比较方向。"""
    jl = chain.get_joints_and_child_links()
    # q=0 做 FK, 拿每个 child link frame 的世界旋转
    n = len(chain.get_joint_parameter_names())
    ret = chain.forward_kinematics(torch.zeros(n))

    items = []
    for joint, child_links in jl:
        if joint.joint_type in ("revolute", "continuous", "prismatic"):
            if "finger" in joint.name.lower():   # 排除夹爪 finger joint
                continue
            child = child_links[0].name
            local_axis = np.asarray(joint.axis).flatten()[:3]
            if child in ret:
                R = ret[child].get_matrix()[0, :3, :3].numpy()
                axis_world = R @ local_axis        # 转到世界座标
            else:
                axis_world = local_axis
            items.append((joint.name, axis_world, child))
    items.reverse()   # 转成 base -> 末端
    return items


def detect(urdf_path):
    with open(urdf_path, "rb") as f:
        chain = pk.build_chain_from_urdf(f.read())
    joints = ordered_joints(chain)
    if not joints:
        return None, "no movable joints"
    names = [j[0] for j in joints]
    axes  = [j[1] for j in joints]
    links = [j[2] for j in joints]

    res = {}
    # shoulder: 第一个不平行于 base
    sh = next((i for i,ax in enumerate(axes) if not is_parallel(ax, BASE_AXIS)), None)
    if sh is None: return None, "no shoulder"
    res["shoulder"] = (names[sh], links[sh])
    # elbow: shoulder 后第一个平行于 shoulder
    el = next((i for i in range(sh+1,len(axes)) if is_parallel(axes[i],axes[sh])), None)
    if el is None: return res, "no elbow"
    res["elbow"] = (names[el], links[el])
    # wrist: elbow 后第一个不平行于 elbow
    wr = next((i for i in range(el+1,len(axes)) if not is_parallel(axes[i],axes[el])), None)
    if wr is None: return res, "no wrist"
    res["wrist"] = (names[wr], links[wr])
    return res, "ok"


def main():
    for name, urdf, manual in ROBOTS:
        print("="*70)
        print(f"[{name}]  {urdf}")
        if not os.path.isfile(urdf):
            print("  [skip] URDF 不存在"); continue
        res, status = detect(urdf)
        print(f"  status: {status}")
        if not res: continue
        # 先印完整关节轴列表 (方便你看规则怎么判的)
        with open(urdf,"rb") as f:
            joints = ordered_joints(pk.build_chain_from_urdf(f.read()))
        print("  运动学链 (base->末端) 各关节轴:")
        for jn, ax, ln in joints:
            print(f"    {jn:16s} axis={np.round(ax,3)}  child={ln}")
        print()
        print(f"  {'role':10s} {'auto joint / link':34s} {'manual':14s} match")
        for role in ["shoulder","elbow","wrist"]:
            a = res.get(role); man = manual.get(role,"-")
            if a:
                jn, ln = a
                m = "OK" if ln==man else "MISMATCH"
                print(f"  {role:10s} {jn+' / '+ln:34s} {man:14s} {m}")
            else:
                print(f"  {role:10s} {'(not found)':34s} {man:14s}")
    print("="*70)


if __name__ == "__main__":
    main()