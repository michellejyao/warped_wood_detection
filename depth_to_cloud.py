import numpy as np
import cv2
import time
from constants import (
    WOOD_PANEL_DEPTH_PATH,   # 16-bit masked depth PNG produced by extract_wood.py
    POINT_CLOUD_PATH,        # where to save the PLY
    # Intrinsics MUST correspond to the aligned depth (same size as the RGB stream after setDepthAlign)
    OAK_D_LITE_INTRINSICS,   # dict: {"fx":..., "fy":..., "cx":..., "cy":...}
)

# Optional knobs
DEPTH_UNITS_MM_TO_M = 1e-3   # OAK-D StereoDepth gives millimeters in uint16 by default -> meters
DOWNSAMPLE_STRIDE = 2        # 1 = full res; 2 = take every other pixel; tune for speed vs size
Z_MIN_M = 0.2                # filter too-near values (meters); adjust to your scene
Z_MAX_M = 10.0               # filter too-far values (meters)

def depth_to_points(depth_u16, fx, fy, cx, cy, stride=1):
    """Back-project a 16-bit depth image (mm) to 3D points in meters (camera frame)."""
    if depth_u16.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth; got {depth_u16.dtype}")

    # Subsample for speed if desired
    depth = depth_u16[::stride, ::stride].astype(np.float32)
    H, W = depth.shape

    # Pixel grid (u,v)
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # shape (H,W)

    # Convert mm -> meters
    Z = depth * DEPTH_UNITS_MM_TO_M

    # Ignore invalids (zeros) early
    valid = Z > 0

    # Back-project: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
    # NOTE: cx,cy must be scaled if using stride != 1. We downsample pixels; intrinsics stay in original scale.
    # Since we took every 'stride' pixel, the pixel coordinates changed by stride. Scale cx,cy accordingly:
    cx_s = cx / stride
    cy_s = cy / stride
    fx_s = fx / stride
    fy_s = fy / stride

    X = (uu - cx_s) * (Z / fx_s)
    Y = (vv - cy_s) * (Z / fy_s)

    # Stack and filter
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    msk = valid.reshape(-1)

    # Range filter (optional)
    msk &= (pts[:, 2] >= Z_MIN_M) & (pts[:, 2] <= Z_MAX_M)

    return pts[msk]

def save_ply(points_xyz, ply_path):
    """Save Nx3 float32 points to ASCII PLY."""
    pts = points_xyz.astype(np.float32)
    N = pts.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(ply_path, "w") as f:
        f.write(header)
        np.savetxt(f, pts, fmt="%.6f")

def main():
    t0 = time.time()
    print("=" * 50)
    print("Depth → 3D point cloud")
    print("=" * 50)

    # 1) Load masked 16-bit depth
    print("\n1. Loading masked depth (16-bit)...")
    depth = cv2.imread(WOOD_PANEL_DEPTH_PATH, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth from {WOOD_PANEL_DEPTH_PATH}")
    if depth.dtype != np.uint16:
        raise ValueError(f"Expected 16-bit depth at {WOOD_PANEL_DEPTH_PATH}, got {depth.dtype}")
    H, W = depth.shape
    print(f"✓ Depth shape: {depth.shape}, dtype={depth.dtype}")

    # 2) Intrinsics (MUST match aligned depth/RGB stream dimensions)
    intr = OAK_D_LITE_INTRINSICS
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    print(f"✓ Using intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # 3) Back-project to 3D (camera frame)
    print("\n2. Back-projecting to 3D...")
    points = depth_to_points(depth, fx, fy, cx, cy, stride=DOWNSAMPLE_STRIDE)
    if points.size == 0:
        print("✗ No valid 3D points (all zeros or filtered).")
        return
    print(f"✓ 3D points: {points.shape[0]}")

    # 4) Save PLY
    print("\n3. Saving PLY...")
    save_ply(points, POINT_CLOUD_PATH)
    print(f"✓ Saved point cloud: {POINT_CLOUD_PATH}")

    print(f"\nDone in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
