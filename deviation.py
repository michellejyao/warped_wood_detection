import numpy as np
import time
from constants import POINT_CLOUD_PATH, DEVIATIONS_PATH, DEVIATION_THRESHOLD

# Load point cloud from PLY file
def load_ply(filename):
    points = []
    with open(filename, 'r') as f:
        header = True
        for line in f:
            if header:
                if line.strip() == 'end_header':
                    header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                x, y, z = map(float, parts[:3])
                points.append([x, y, z])
    return np.array(points, dtype=np.float32)

# Fit plane z = a x + b y + c (least squares)
def fit_plane(points):
    X = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    Z = points[:, 2]
    coeffs, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    return coeffs  # (a, b, c)

def compute_deviations(points, plane_coeffs):
    a, b, c = plane_coeffs
    z_plane = a * points[:, 0] + b * points[:, 1] + c
    return points[:, 2] - z_plane

# Robust plane via RANSAC
def fit_plane_ransac(points, distance_threshold=0.002, max_iters=500, min_inliers_ratio=0.5, random_state=42):
    """
    distance_threshold in meters (e.g., 0.002 = 2 mm)
    Returns: best_coeffs, inlier_mask (bool)
    """
    rng = np.random.default_rng(random_state)
    n = points.shape[0]
    if n < 3:
        return fit_plane(points), np.ones(n, dtype=bool)

    best_inliers = np.zeros(n, dtype=bool)
    best_coeffs = None

    for _ in range(max_iters):
        idx = rng.choice(n, size=3, replace=False)
        sample = points[idx]
        coeffs = fit_plane(sample)
        dev = compute_deviations(points, coeffs)
        inliers = np.abs(dev) <= distance_threshold

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_coeffs = fit_plane(points[inliers])

            if inliers.sum() >= int(min_inliers_ratio * n):
                # Good enough; early stop
                break

    if best_coeffs is None:
        best_coeffs = fit_plane(points)
        best_inliers = np.ones(n, dtype=bool)
    return best_coeffs, best_inliers

if __name__ == "__main__":
    t0 = time.time()

    ply_file = POINT_CLOUD_PATH
    deviation_threshold = DEVIATION_THRESHOLD

    points = load_ply(ply_file)
    if points.shape[0] == 0:
        print("No points loaded from point cloud.")
        exit(1)

    # Optional: prefilter invalids
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] < 3:
        print("Not enough valid points.")
        exit(1)

    # RANSAC fit
    plane_coeffs, inliers = fit_plane_ransac(points, distance_threshold=0.002, max_iters=800, min_inliers_ratio=0.6)
    print(f"Fitted (RANSAC) plane: z = {plane_coeffs[0]:.6f}*x + {plane_coeffs[1]:.6f}*y + {plane_coeffs[2]:.6f}")
    print(f"Inliers: {inliers.sum()}/{len(inliers)}")

    # Deviations on inliers only
    deviations = compute_deviations(points[inliers], plane_coeffs)
    std_dev = np.std(deviations)
    print(f"Std dev of vertical deviations (inliers): {std_dev:.6f} m")

    # Save deviations (for analysis)
    np.savetxt(DEVIATIONS_PATH, deviations)

    # Decision
    if std_dev > deviation_threshold:
        print(f"Wood panel is WARPED (std dev > {deviation_threshold})")
    else:
        print(f"Wood panel is FLAT (std dev <= {deviation_threshold})")

    print(f"Execution time: {time.time() - t0:.2f} s")
