"""Multi-view triangulation of object keypoints -> 3D position in the robot base / world frame.

Camera model = gen_vision_data.project (MuJoCo: the camera looks along its -z axis, +y is image-up, so v grows
downwards):  u = cx + f * x_c / depth,  v = cy - f * y_c / depth,  depth = -z_c,  p_c = R^T (p - cam_pos).

triangulate(uv, w, cams)        weighted linear least squares (DLT) over the views that are usable, then a few
                                Gauss-Newton steps on the pixel reprojection error -> the maximum-likelihood
                                point for pixel noise sigma_i ~ 1 / sqrt(w_i)
fallback with ONE good view     intersect its ray with the plane z = plane_z (the cube resting on the table,
                                the tape on the table); without a plane the point is not observable -> None
outliers                        a view whose reprojection error is > max_px after the fit is dropped and the fit redone
"""
import numpy as np


def ray(uv, K, cam_pos, cam_xmat):
    """pixel -> (origin, unit direction) in the world"""
    R = np.asarray(cam_xmat).reshape(3, 3)
    x = (uv[0] - K[0, 2]) / K[0, 0]
    y = -(uv[1] - K[1, 2]) / K[1, 1]
    dc = np.array([x, y, -1.0])                      # camera frame: looks along -z
    d = R @ dc
    return np.asarray(cam_pos, float), d / np.linalg.norm(d)


def project(p, K, cam_pos, cam_xmat):
    R = np.asarray(cam_xmat).reshape(3, 3)
    pc = R.T @ (np.asarray(p) - cam_pos)
    depth = -pc[2]
    return np.array([K[0, 2] + K[0, 0] * pc[0] / depth, K[1, 2] - K[1, 1] * pc[1] / depth]), depth


def _P(K, cam_pos, cam_xmat):
    """3x4 projection matrix in the same convention (v flipped) for the linear (DLT) solve"""
    R = np.asarray(cam_xmat).reshape(3, 3)
    A = np.diag([1.0, -1.0, -1.0]) @ R.T              # world -> (x, -y, depth) camera coordinates
    t = -A @ np.asarray(cam_pos)
    return K @ np.hstack([A, t[:, None]])


def _reproj(p, uv, cams):
    return np.array([np.linalg.norm(project(p, *c)[0] - u) for u, c in zip(uv, cams)])


def closest_on_ray(o, d, p):
    """point on the ray o + t d (t >= 0) closest to p"""
    return o + max(0.0, float(np.dot(np.asarray(p) - o, d))) * d


def triangulate(uv, w, cams, plane_z=None, min_w=0.3, max_px=8.0, iters=5, prior=None, max_jump=0.03):
    """
    uv      (V, 2) pixel keypoints, one per camera
    w       (V,)   confidence per view (heatmap confidence); views below min_w are not used
    cams    list of (K, cam_pos, cam_xmat) per view
    plane_z height of the plane the object rests on (single-view fallback); None = no fallback
    prior   last known 3D position (e.g. the previous frame): when the views disagree / only one is usable and
            there is no plane (object in the hand), take the point on the most confident view's ray closest to it.
            With two views an error ALONG the epipolar line gives no reprojection error, so a fit that jumps more
            than max_jump [m] from the prior is also treated as a disagreement.
    returns xyz (3,) or None, info dict {views: used view ids, err: reprojection px per used view, mode}
    """
    uv = np.asarray(uv, float); w = np.asarray(w, float)
    use = [i for i in range(len(cams)) if w[i] >= min_w and np.all(np.isfinite(uv[i]))]
    for _ in range(len(cams)):
        if len(use) >= 2:
            rows = []
            for i in use:
                P = _P(*cams[i]); s = np.sqrt(w[i])
                rows += [s * (uv[i, 0] * P[2] - P[0]), s * (uv[i, 1] * P[2] - P[1])]
            X = np.linalg.svd(np.array(rows))[2][-1]
            p = X[:3] / X[3]
            for _ in range(iters):                    # Gauss-Newton on the weighted pixel error
                J, r = [], []
                for i in use:
                    f0, _ = project(p, *cams[i]); s = np.sqrt(w[i])
                    Jp =np.stack([(project(p + np.eye(3)[k] * 1e-5, *cams[i])[0] - f0) / 1e-5 for k in range(3)], 1)
                    J.append(s * Jp); r.append(s * (f0 - uv[i]))
                J = np.concatenate(J, 0); r = np.concatenate(r)
                p = p - np.linalg.lstsq(J, r, rcond=None)[0]
            err = _reproj(p, uv[use], [cams[i] for i in use])
            jump = prior is not None and np.linalg.norm(p - np.asarray(prior)) > max_jump
            if jump and len(use) == 2:
                err = np.maximum(err, max_px + 1)                       # consistent views but an impossible jump
            if err.max() <= max_px or len(use) == 2 and plane_z is None and prior is None:
                return p, {"views": use, "err": err, "mode": "triangulated"}
            if len(use) == 2:                         # two views disagree: keep the more confident one
                use = [use[int(np.argmax(w[use]))]]
                continue
            use = [u for u, e in zip(use, err) if e < err.max()]   # drop the worst view, refit
            continue
        if len(use) == 1 and plane_z is None and prior is not None:
            o, d = ray(uv[use[0]], *cams[use[0]])
            return closest_on_ray(o, d, prior), {"views": use, "err": np.zeros(1), "mode": "prior"}
        if len(use) == 1 and plane_z is not None:
            o, d = ray(uv[use[0]], *cams[use[0]])
            if abs(d[2]) < 1e-6:
                return None, {"views": use, "err": None, "mode": "none"}
            t = (plane_z - o[2]) / d[2]
            return o + t * d, {"views": use, "err": np.zeros(1), "mode": "plane"}
        break
    return None, {"views": use, "err": None, "mode": "none"}


# ------------------------------------------------------------------------------------------------ self test
if __name__ == "__main__":
    import mujoco

    def look_at(pos, target):
        z = np.asarray(pos) - target; z /= np.linalg.norm(z)
        x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
        return np.column_stack([x, y, z]).flatten()

    W, Hh, fovy = 768, 432, 42.0
    f = 0.5 * Hh / np.tan(np.radians(fovy) / 2)
    K = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.0]])
    C = np.array([0.475, 0.025, 0.0])
    cams = [(K, np.array([0.85, 0.65, 0.50]), look_at([0.85, 0.65, 0.50], C)),
            (K, np.array([0.85, -0.65, 0.50]), look_at([0.85, -0.65, 0.50], C))]
    rng = np.random.default_rng(0)
    errs, errs1, errs_plane = [], [], []
    for _ in range(2000):
        p = np.array([rng.uniform(0.30, 0.65), rng.uniform(-0.30, 0.35), 0.025])
        uv = np.stack([project(p, *c)[0] for c in cams])
        noisy = uv + rng.normal(0, 1.0, uv.shape)                      # 1 px keypoint noise
        q, info = triangulate(noisy, [1, 1], cams, plane_z=0.025)
        errs.append(np.linalg.norm(q - p) * 1000)
        q1, _ = triangulate(noisy, [1, 0], cams, plane_z=0.025)        # camera B occluded -> plane fallback
        errs_plane.append(np.linalg.norm(q1 - p) * 1000)
        bad = noisy.copy(); bad[1] += 40                               # camera B wrong by 40 px (occluder)
        q2, info2 = triangulate(bad, [1, 0.9], cams, plane_z=0.025)
        errs1.append(np.linalg.norm(q2 - p) * 1000)
    print(f"1 px noise, both views:          3D error median {np.median(errs):.2f} mm, p95 {np.percentile(errs, 95):.2f} mm")
    print(f"1 px noise, one view + table:    3D error median {np.median(errs_plane):.2f} mm, p95 {np.percentile(errs_plane, 95):.2f} mm")
    print(f"view B off by 40 px (rejected):  3D error median {np.median(errs1):.2f} mm, p95 {np.percentile(errs1, 95):.2f} mm")
    e3, e4 = [], []                                   # in the hand (no plane): view A wrong by 60 px, less confident
    for _ in range(2000):
        p = np.array([rng.uniform(0.30, 0.65), rng.uniform(-0.30, 0.35), rng.uniform(0.05, 0.20)])
        uv = np.stack([project(p, *c)[0] for c in cams]) + rng.normal(0, 1.0, (2, 2)); uv[0] += 60
        q0, _ = triangulate(uv, [0.6, 0.8], cams)
        q1, _ = triangulate(uv, [0.6, 0.8], cams, prior=p + rng.normal(0, 0.005, 3))   # prior 5 mm off
        e3.append(np.linalg.norm(q0 - p) * 1000); e4.append(np.linalg.norm(q1 - p) * 1000)
    print(f"in hand, view A off by 60 px:    no prior {np.median(e3):.1f} mm | with last position as prior "
          f"{np.median(e4):.1f} mm (p95 {np.percentile(e4, 95):.1f})")