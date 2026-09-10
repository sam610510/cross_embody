import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 把腳本的上一層目錄 (cross_embody) 加到 Python 路徑中 ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from fk_encoder.fk_encoder import FKEncoder


def verify_scale_invariance(encoder_normal, encoder_scaled, num_tests=10):
    """
    驗證尺度不變性：
    要求 encoder_scaled 的 URDF 是 encoder_normal 的等比例放大/縮小版。
    """
    print("=== 1. 驗證尺度不變性 (Scale Invariance) ===")

    torch.manual_seed(42)
    q_test = torch.rand((num_tests, encoder_normal.arm_dof)) * 2 - 1  # [-1, 1] rad

    out_norm = encoder_normal(q_test)
    out_scal = encoder_scaled(q_test)

    # 21維輸出: p_ee 佔 0:9 (position 3 + 6D rot 6), v_kp 佔 9:18
    vkp_norm = out_norm[:, 9:18]
    vkp_scal = out_scal[:, 9:18]

    max_diff = torch.max(torch.abs(vkp_norm - vkp_scal)).item()
    print(f"隨機 {num_tests} 組姿態下的最大 v_kp 誤差: {max_diff:.8f}")

    if torch.allclose(vkp_norm, vkp_scal, atol=1e-5):
        print("✅ 測試通過：v_kp 在 Link 長度變化時保持高度一致。")
    else:
        print("❌ 測試失敗：v_kp 數值發生偏移，請檢查 URDF 的等比例縮放或 L 的計算。")
    print("-" * 50)


def verify_q0_calibration(encoder):
    """
    驗證 q=0 時的 keyvectors。
    """
    print("=== 2. 校正 q=0 時的 Keyvectors ===")
    q0 = torch.zeros((1, encoder.arm_dof))
    out = encoder(q0)

    # 21維輸出: p_ee 佔 0:9, v_kp 佔 9:18
    v_kp = out[0, 9:18].numpy()
    v_SE = v_kp[0:3]
    v_SW = v_kp[3:6]
    v_SEE = v_kp[6:9]

    print("在 q=0 的狀態下：")
    print(f"上臂長度 L (正規化基準): {encoder.L.item():.4f} m")
    print(f"v_SE  (Elbow relative to Shoulder): {v_SE} -> 長度應為 1.0 (目前: {np.linalg.norm(v_SE):.4f})")
    print(f"v_SW  (Wrist relative to Shoulder): {v_SW}")
    print(f"v_SEE (EE relative to Shoulder)   : {v_SEE}")

    assert np.isclose(np.linalg.norm(v_SE), 1.0, atol=1e-5), "v_SE 長度不等於 1！"
    print("✅ 基準向量擷取正常，請核對上述方向向量是否符合您的 URDF 預設基準姿態 (Home Position)。")
    print("-" * 50)


def verify_smoothness_and_monotonicity(encoder, joint_idx=0, steps=200):
    """
    驗證單調性與平滑度：
    固定其他關節為 BASE_POSE，將目標關節從 -pi/2 掃描到 pi/2，
    並將每個向量分別存成獨立圖表。
    """
    print(f"=== 3. 驗證平滑性 (Sweep Joint {joint_idx}) ===")

    # 基準姿態（與 render_mujoco 的 BASE_POSE 一致）
    BASE_POSE = torch.tensor([0.0, -0.5, 0.0, -1.8, 0.0, 1.2, 0.0])

    # 掃描範圍：−π/2 → π/2（與 MuJoCo gif 對齊）
    SWEEP = np.pi / 2
    angles = torch.linspace(-SWEEP, SWEEP, steps)

    q_sweep = BASE_POSE.repeat(steps, 1)
    q_sweep[:, joint_idx] = angles

    out = encoder(q_sweep)
    v_kp = out[:, 9:18]  # 21維: p_ee 佔 0:9, v_kp 佔 9:18

    diffs = torch.diff(v_kp, dim=0)
    max_jump = torch.max(torch.abs(diffs)).item()

    print(f"掃描過程中，單步最大向量突變量: {max_jump:.6f}")
    threshold = (2 * SWEEP / steps) * 2.0
    if max_jump < threshold:
        print("✅ 測試通過：v_kp 變化平滑連續，無奇異點突變。")
    else:
        print("⚠️ 警告：偵測到較大的數值跳變，請檢查圖表是否有不連續現象。")

    # --- 繪圖視覺化 (拆分三張獨立圖表) ---
    angles_np = angles.numpy()
    v_kp_np = v_kp.numpy()
    labels = ['X', 'Y', 'Z']

    vectors = [
        ("v_SE",  v_kp_np[:, 0:3], "Elbow relative to Shoulder"),
        ("v_SW",  v_kp_np[:, 3:6], "Wrist relative to Shoulder"),
        ("v_SEE", v_kp_np[:, 6:9], "End-Effector relative to Shoulder"),
    ]

    pi_ticks = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
    pi_labels = [r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$']

    for name, vec_data, physical_meaning in vectors:
        fig, ax = plt.subplots(figsize=(10, 6))

        for j in range(3):
            ax.plot(angles_np, vec_data[:, j], label=f'{name}_{labels[j]}', linewidth=2.5)

        ax.set_title(f"{name} Spatial Components\n({physical_meaning})",
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel("Normalized Value\n(Actual Distance / Upper Arm Length L)",
                      fontsize=12, fontweight='bold')
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_labels, fontsize=14)
        ax.set_xlabel(f"Joint {joint_idx} Angle (Radians)", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title="Components")
        plt.tight_layout()

        filename = f"{name}_smoothness_labeled.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ 已儲存獨立圖表：{filename}")

    print("-" * 50)


if __name__ == "__main__":
    yaml_path = "configs/robots.yaml"

    try:
        encoder_normal = FKEncoder(robot_name="fr3", robots_yaml=yaml_path)

        verify_q0_calibration(encoder_normal)
        verify_smoothness_and_monotonicity(encoder_normal, joint_idx=0)

        try:
            encoder_scaled = FKEncoder(robot_name="fr3_scaled", robots_yaml=yaml_path)
            verify_scale_invariance(encoder_normal, encoder_scaled)
        except KeyError:
            print("提示：若要測試尺度不變性，請在 configs/robots.yaml 建立一個 'fr3_scaled' 的 entry。")

    except Exception as e:
        import traceback
        print(f"初始化或測試失敗: {e}")
        traceback.print_exc()