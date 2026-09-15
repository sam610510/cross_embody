"""
Damped least-squares differential IK,用tcp site當end-effector參考點。
只解位置(3-DOF)+ z軸朝下的orientation,對pick task夠用,不用求完整6-DOF姿態最佳化。
"""
import mujoco
import numpy as np

N_ARM_JOINTS = 7


def solve_ik(model, data, target_pos, target_quat, q_init,
             site_name="tcp", max_iters=200, tol=1e-4, damping=1e-2, step_scale=1.0):
    """回傳收斂後的7維手臂關節角。q_init是warm-start,避免IK跳到奇怪的解(呼應你FK decoder那邊對IK多解消歧義的顧慮)。"""
    data.qpos[:N_ARM_JOINTS] = q_init
    mujoco.mj_forward(model, data)
    site_id = model.site(site_name).id

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        cur_pos = data.site_xpos[site_id].copy()
        cur_mat = data.site_xmat[site_id].reshape(3, 3)
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, cur_mat.flatten())

        pos_err = target_pos - cur_pos

        quat_err = np.zeros(3)
        neg_cur = np.zeros(4)
        mujoco.mju_negQuat(neg_cur, cur_quat)
        err_quat = np.zeros(4)
        mujoco.mju_mulQuat(err_quat, target_quat, neg_cur)
        mujoco.mju_quat2Vel(quat_err, err_quat, 1.0)

        err = np.concatenate([pos_err, 0.3 * quat_err])  # orientation權重降低,位置優先收斂
        if np.linalg.norm(pos_err) < tol and np.linalg.norm(quat_err) < tol * 5:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.concatenate([jacp[:, :N_ARM_JOINTS], jacr[:, :N_ARM_JOINTS]], axis=0)

        JJt = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        data.qpos[:N_ARM_JOINTS] += step_scale * dq
        data.qpos[:N_ARM_JOINTS] = np.clip(
            data.qpos[:N_ARM_JOINTS],
            model.jnt_range[:N_ARM_JOINTS, 0], model.jnt_range[:N_ARM_JOINTS, 1],
        )

    return data.qpos[:N_ARM_JOINTS].copy()