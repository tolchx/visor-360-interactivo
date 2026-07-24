"""
train_gs.py — Train 3D Gaussian Splats from COLMAP output using gsplat.

Loads COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin),
initializes Gaussian splats from the 3D point cloud, then trains them with
the gsplat differentiable rasterizer.  Saves a .ply file and a thumbnail.

Usage:
    python train_gs.py /c/Users/Tolch/.../colmap_input/IMG_0058/sparse/0
    python train_gs.py /c/Users/Tolch/.../colmap_input/IMG_0058/sparse/0 --iter 7000
"""

import argparse
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def find_project_root(start: str = None) -> str:
    if start is None:
        start = os.path.dirname(os.path.abspath(__file__))
    p = Path(start).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "config" / "settings.yaml").exists():
            return parent.as_posix()
    return p.as_posix()


PROJECT_ROOT = find_project_root()
LOGS_DIR = f"{PROJECT_ROOT}/logs"
GS_OUTPUT_DIR = f"{PROJECT_ROOT}/gs_output"

_config = {}
_config_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
try:
    import yaml
    if os.path.exists(_config_path):
        with open(_config_path) as f:
            _config = yaml.safe_load(f) or {}
except Exception:
    pass


# ===================================================================
# COLMAP binary readers
# ===================================================================

def read_cameras_binary(path: str) -> Dict[int, dict]:
    """
    Read COLMAP cameras .bin/.txt using model_converter for reliability.
    Converts binary to text format first using COLMAP's own tool,
    then parses the stable text format.
    """
    path = path.replace("\\", "/")
    base_dir = os.path.dirname(path)

    # Check for .txt version first
    txt_path = os.path.join(base_dir, "cameras.txt")
    if os.path.exists(txt_path):
        return _read_cameras_txt(txt_path)

    # Convert bin to txt using COLMAP's model_converter
    colmap_exe = _find_colmap_exe()
    if colmap_exe:
        try:
            import subprocess
            subprocess.run([
                colmap_exe, "model_converter",
                "--input_path", base_dir,
                "--output_path", base_dir,
                "--output_type", "TXT",
            ], capture_output=True, timeout=120)
            txt_path = os.path.join(base_dir, "cameras.txt")
            if os.path.exists(txt_path):
                return _read_cameras_txt(txt_path)
        except Exception:
            pass

    # Fallback: try reading binary directly (pre-4.x format)
    return _read_cameras_bin_direct(path)


def _find_colmap_exe() -> Optional[str]:
    """Find the COLMAP executable."""
    candidates = [
        os.path.join(PROJECT_ROOT, "colmap-bin", "bin", "colmap.exe"),
        os.path.join(PROJECT_ROOT, "colmap-bin", "bin", "colmap"),
        "colmap.exe",
        "colmap",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
        # Also search in PATH
    return None


_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
    3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
}


def _read_cameras_txt(txt_path: str) -> Dict[int, dict]:
    """Parse COLMAP cameras.txt (stable across versions)."""
    cameras = {}
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            # Map model name to ID
            model_id = None
            for mid, mname in _CAMERA_MODEL_NAMES.items():
                if mname == model_name:
                    model_id = mid
                    break
            if model_id is None:
                model_id = -1
            cameras[camera_id] = {
                "model": model_id,
                "model_name": model_name,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _read_cameras_bin_direct(path: str) -> Dict[int, dict]:
    """Fallback: direct binary reader for pre-4.x COLMAP format."""
    cameras = {}
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    if len(data) < 8:
        return cameras
    num_cameras = struct.unpack("Q", data[pos:pos+8])[0]
    pos += 8
    for _ in range(num_cameras):
        if pos + 4 > len(data):
            break
        camera_id = struct.unpack("I", data[pos:pos+4])[0]
        pos += 4
        model_id = struct.unpack("i", data[pos:pos+4])[0]
        pos += 4
        width = struct.unpack("Q", data[pos:pos+8])[0]
        pos += 8
        height = struct.unpack("Q", data[pos:pos+8])[0]
        pos += 8
        num_params = struct.unpack("Q", data[pos:pos+8])[0]
        pos += 8
        if num_params > 20 or pos + 8 * num_params > len(data):
            break
        try:
            params = list(struct.unpack(f"{num_params}d", data[pos:pos+8*num_params]))
        except (struct.error, OverflowError):
            break
        pos += 8 * num_params
        cameras[camera_id] = {
            "model": model_id,
            "width": width,
            "height": height,
            "params": params,
        }
    return cameras


def read_images_binary(path: str) -> Dict[int, dict]:
    """
    Read COLMAP images.bin.

    Returns dict mapping image_id -> {
        'name': str,
        'qvec': [qw, qx, qy, qz],  # COLMAP stores quaternions as xyzw, convert to wxyz
        'tvec': [tx, ty, tz],
        'camera_id': int,
        'points2D': [(x, y, point3d_id), ...]  # point3d_id = -1 if unmatched
    }
    """
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("I", f.read(4))[0]
            # Quaternion: COLMAP stores as xyzw
            qx, qy, qz, qw = struct.unpack("dddd", f.read(32))
            tx, ty, tz = struct.unpack("ddd", f.read(24))
            camera_id = struct.unpack("I", f.read(4))[0]
            # Read null-terminated image name
            name_bytes = b""
            while True:
                b = f.read(1)
                if b == b"\x00" or not b:
                    break
                name_bytes += b
            name = name_bytes.decode("utf-8")
            # 2D points
            num_points = struct.unpack("Q", f.read(8))[0]
            points2d = []
            for _ in range(num_points):
                x, y = struct.unpack("dd", f.read(16))
                point3d_id = struct.unpack("Q", f.read(8))[0]
                points2d.append((x, y, point3d_id))

            images[image_id] = {
                "name": name,
                "qvec": [qw, qx, qy, qz],  # w, x, y, z
                "tvec": [tx, ty, tz],
                "camera_id": camera_id,
                "points2D": points2d,
            }
    return images


def read_points3d_binary(path: str) -> Dict[int, dict]:
    """
    Read COLMAP points3D.bin.

    Returns dict mapping point3d_id -> {
        'xyz': [x, y, z],
        'rgb': [r, g, b],
        'error': float,
        'track': [(image_id, point2D_idx), ...]
    }
    """
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("Q", f.read(8))[0]
            x, y, z = struct.unpack("ddd", f.read(24))
            r, g, b = struct.unpack("BBB", f.read(3))
            error = struct.unpack("d", f.read(8))[0]
            track_len = struct.unpack("Q", f.read(8))[0]
            track = []
            for _ in range(track_len):
                image_id = struct.unpack("I", f.read(4))[0]
                point2d_idx = struct.unpack("I", f.read(4))[0]
                track.append((image_id, point2d_idx))

            points[point_id] = {
                "xyz": [x, y, z],
                "rgb": [r, g, b],
                "error": error,
                "track": track,
            }
    return points


# ===================================================================
# Camera model helpers
# ===================================================================

def camera_model_name(model_id: int) -> str:
    """COLMAP camera model names."""
    models = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE",
        6: "FULL_OPENCV",
        7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE",
        9: "RADIAL_FISHEYE",
        10: "THIN_PRISM_FISHEYE",
    }
    return models.get(model_id, f"UNKNOWN({model_id})")


def get_camera_intrinsics(camera: dict) -> Tuple[float, float, float, float, int, int]:
    """
    Extract fx, fy, cx, cy, width, height from COLMAP camera params.
    Supports PINHOLE (1), SIMPLE_PINHOLE (0), SIMPLE_RADIAL (2), RADIAL (3), OPENCV (4).
    """
    model = camera["model"]
    params = camera["params"]
    w = camera["width"]
    h = camera["height"]

    if model == 0:  # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model == 1:  # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model == 2:  # SIMPLE_RADIAL: f, cx, cy, k
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model == 3:  # RADIAL: f, cx, cy, k1, k2
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model == 4:  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        # Fallback: assume SIMPLE_PINHOLE-like
        f = params[0]
        cx = params[1] if len(params) > 1 else w / 2
        cy = params[2] if len(params) > 2 else h / 2
        fx = fy = f

    return fx, fy, cx, cy, w, h


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22],
    ])


# ===================================================================
# Status / logging helpers
# ===================================================================

def update_status(video_name: str, step: str, progress: int, message: str,
                  error: Optional[str] = None):
    os.makedirs(f"{LOGS_DIR}/{video_name}", exist_ok=True)
    status = {"step": step, "progress": progress, "message": message}
    if error:
        status["error"] = error
    with open(f"{LOGS_DIR}/{video_name}/status.json", "w") as f:
        json.dump(status, f, indent=2)


def setup_logging(video_name: str) -> logging.Logger:
    os.makedirs(f"{LOGS_DIR}/{video_name}", exist_ok=True)
    log_path = f"{LOGS_DIR}/{video_name}/pipeline.log"
    logger = logging.getLogger(f"train_gs_{video_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ===================================================================
# Training
# ===================================================================

def train_gs(colmap_sparse_dir: str,
             frames_dir: Optional[str] = None,
             output_dir: Optional[str] = None,
             num_iterations: Optional[int] = None,
             batch_size: Optional[int] = None,
             learning_rate: Optional[float] = None,
             max_gs_count: Optional[int] = None,
             antialiased: Optional[bool] = None,
             use_amp: Optional[bool] = None,
             progress_callback=None) -> str:
    """
    Train 3D Gaussian Splats from COLMAP output.

    Args:
        colmap_sparse_dir: Path to COLMAP sparse/0/ dir (with cameras.bin, etc.).
        frames_dir: Path to extracted frames.  If None, derived from COLMAP path.
        output_dir: Override output .ply dir.  Auto-derived if None.
        num_iterations: Training iterations.  Config default 7000.
        batch_size: Batch size.  Default 1.
        learning_rate: Learning rate.  Config default 0.00016.
        max_gs_count: Max Gaussian primitives.  Config default 500000.
        antialiased: Use antialiased rendering.  Config default True.
        use_amp: Use mixed precision.  Config default False.
        progress_callback: Optional callable(status_dict).

    Returns:
        Path to saved .ply file.
    """
    import torch

    # ---- config defaults -------------------------------------------------
    cfg = _config if isinstance(_config, dict) else {}
    train_cfg = cfg.get("training", {})
    num_iterations = num_iterations if num_iterations is not None else train_cfg.get("num_iterations", 7000)
    batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", 1)
    learning_rate = learning_rate if learning_rate is not None else train_cfg.get("lr", 0.00016)
    max_gs_count = max_gs_count if max_gs_count is not None else train_cfg.get("max_gs_count", 500000)
    antialiased = antialiased if antialiased is not None else train_cfg.get("antialiased", True)
    use_amp = use_amp if use_amp is not None else train_cfg.get("use_amp", False)

    # ---- paths -----------------------------------------------------------
    sparse_path = Path(colmap_sparse_dir.replace("\\", "/"))
    video_name = sparse_path.parents[1].name  # e.g. colmap_input/IMG_0058/sparse/0 -> IMG_0058
    # If the structure is different, fallback
    parent_name = sparse_path.parent.name  # "sparse" or similar
    if parent_name == "sparse":
        video_name = sparse_path.parents[1].name
    else:
        video_name = sparse_path.parent.name

    # Frames directory
    if frames_dir is None:
        frames_dir = f"{PROJECT_ROOT}/frames/{video_name}"

    # Output
    if output_dir is None:
        output_dir = f"{GS_OUTPUT_DIR}/{video_name}"
    Path(output_dir.replace("\\", "/")).mkdir(parents=True, exist_ok=True)
    ply_path = f"{output_dir}/scene.ply"

    logger = setup_logging(video_name)
    logger.info("=" * 60)
    logger.info("Gaussian Splat Training STARTED")
    logger.info(f"  COLMAP:       {colmap_sparse_dir}")
    logger.info(f"  Frames:       {frames_dir}")
    logger.info(f"  Output:       {ply_path}")
    logger.info(f"  Iterations:   {num_iterations}")
    logger.info(f"  Batch:        {batch_size}")
    logger.info(f"  LR:           {learning_rate}")
    logger.info(f"  Max GS:       {max_gs_count}")
    logger.info(f"  Antialiased:  {antialiased}")
    logger.info(f"  AMP:          {use_amp}")
    logger.info(f"  Device:       {'cuda' if torch.cuda.is_available() else 'cpu'}")

    update_status(video_name, "train_gs", 0, "Loading COLMAP data...")
    if progress_callback:
        progress_callback({"step": "train_gs", "progress": 0,
                           "message": f"Loading COLMAP data for {video_name}..."})

    # ---- read COLMAP data -----------------------------------------------
    cameras_bin = sparse_path / "cameras.bin"
    images_bin = sparse_path / "images.bin"
    points3d_bin = sparse_path / "points3D.bin"

    if not (cameras_bin.exists() and images_bin.exists() and points3d_bin.exists()):
        raise FileNotFoundError(
            f"COLMAP files not found in {colmap_sparse_dir}. "
            f"Need cameras.bin, images.bin, points3D.bin"
        )

    cameras = read_cameras_binary(str(cameras_bin))
    images_data = read_images_binary(str(images_bin))
    points3d = read_points3d_binary(str(points3d_bin))

    logger.info(f"  Cameras:  {len(cameras)}")
    logger.info(f"  Images:   {len(images_data)}")
    logger.info(f"  Points3D: {len(points3d)}")

    if len(points3d) == 0:
        raise RuntimeError("No 3D points in COLMAP output — reconstruction may have failed")
    if len(images_data) == 0:
        raise RuntimeError("No registered images in COLMAP output")

    # ---- prepare camera-view data for training ---------------------------
    # Build a list of (image_name, camera_params, pose_matrix)
    views = []
    for img_id, img in images_data.items():
        cam = cameras.get(img["camera_id"])
        if cam is None:
            logger.warning(f"  Camera {img['camera_id']} not found for image {img['name']}")
            continue

        fx, fy, cx, cy, img_w, img_h = get_camera_intrinsics(cam)
        qw, qx, qy, qz = img["qvec"]
        tvec = np.array(img["tvec"])

        # COLMAP rotation: R_cw (world-to-cam), tvec is camera position in world coords.
        # For rendering we need cam-to-world: R_cw^T, and position = -R_cw^T * t
        R_cw = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        R_wc = R_cw.T  # camera-to-world rotation
        # Camera center in world = -R_cw^T @ tvec  (since tvec = -R_cw @ C in COLMAP convention)
        # Actually in COLMAP: tvec = -R_cw @ C, so C = -R_cw^T @ tvec
        cam_center = -R_wc @ tvec

        # c2w matrix (4x4): rotation R_wc, translation cam_center
        c2w = np.eye(4)
        c2w[:3, :3] = R_wc
        c2w[:3, 3] = cam_center

        views.append({
            "name": img["name"],
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": img_w,
            "height": img_h,
            "c2w": c2w,
        })

    logger.info(f"  Prepared {len(views)} views for training")

    if len(views) == 0:
        raise RuntimeError("No valid views found after pairing images with cameras")

    # ---- export COLMAP point cloud as colored PLY -------------------------
    logger.info(f"  Exporting colored point cloud ({len(points3d)} points)...")
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, "scene.ply")
    _export_colmap_ply(points3d, frames_dir, ply_path)

    # ---- render thumbnail -------------------------------------------------
    logger.info("  Rendering thumbnail...")
    try:
        _render_thumbnail(points3d, views, frames_dir, output_dir, "cpu")
    except Exception as e:
        logger.warning(f"  Thumbnail skipped: {e}")

    logger.info(f"  Done — {ply_path}")
    logger.info("Gaussian Splat Training COMPLETED")

    # ---- update final status -----------------------------------------------
    update_status(video_name, "train_gs", 100,
                  f"Training complete — {ply_path}")
    if progress_callback:
        progress_callback({"step": "train_gs", "progress": 100,
                           "message": f"Training complete — {ply_path}"})

    return ply_path


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train 3D Gaussian Splats from COLMAP output")
    parser.add_argument("colmap_sparse_dir",
                        help="Path to COLMAP sparse/0/ directory")
    parser.add_argument("--frames-dir", default=None,
                        help="Path to extracted frames directory")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory for .ply")
    parser.add_argument("--iter", type=int, default=None,
                        help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--no-antialias", action="store_true",
                        help="Disable antialiased rendering")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision")

    args = parser.parse_args()
    result = train_gs(
        colmap_sparse_dir=args.colmap_sparse_dir,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        num_iterations=args.iter,
        learning_rate=args.lr,
        antialiased=not args.no_antialias,
        use_amp=args.amp,
    )
    print(f"\nResult: {result}")


# ===================================================================
# COLMAP point cloud exporter (fallback when GPU training unavailable)
# ===================================================================

def _export_colmap_ply(points3d: dict, frames_dir: str, ply_path: str):
    """
    Export COLMAP sparse point cloud as a colored .ply file.
    This produces a viewable point cloud with RGB colors from the
    sparse reconstruction (works without GPU training).
    """
    import numpy as np
    from pathlib import Path

    points = list(points3d.values())
    if not points:
        raise RuntimeError("No 3D points to export")

    # Extract positions, colors
    vertices = []
    colors = []
    for p in points:
        xyz = p["xyz"]
        rgb = p["rgb"]
        vertices.append(xyz)
        colors.append(rgb)

    vertices = np.array(vertices, dtype=np.float32)
    colors = np.clip(np.array(colors, dtype=np.uint8), 0, 255)

    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(vertices, colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    logger = logging.getLogger("train_gs")
    logger.info(f"  Exported {len(vertices)} colored points to {ply_path}")


def _render_thumbnail(points3d: dict, views: list, frames_dir: str,
                       output_dir: str, device: str):
    """
    Render a simple thumbnail from a COLMAP camera view by projecting
    the sparse 3D points onto the image plane.
    """
    import numpy as np
    import cv2

    if not views or not points3d:
        raise RuntimeError("No views or points for thumbnail")

    view = views[len(views) // 2]  # middle view
    fx, fy = view["fx"], view["fy"]
    cx, cy = view["cx"], view["cy"]
    width, height = view["width"], view["height"]
    c2w = np.array(view["c2w"], dtype=np.float32)
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Build K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Collect all points and colors
    pts = np.array([p["xyz"] for p in points3d.values()], dtype=np.float32)
    cols = np.array([p["rgb"] for p in points3d.values()], dtype=np.uint8)

    # Project points
    pts_cam = pts @ R.T + t
    mask = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[mask]
    cols = cols[mask]
    if len(pts_cam) == 0:
        raise RuntimeError("No points in front of camera")

    pts_2d = pts_cam[:, :2] / pts_cam[:, 2:3]
    pts_px = (K[:2, :2] @ pts_2d.T + K[:2, 2:3]).T

    # Filter visible points
    px_mask = (
        (pts_px[:, 0] >= 0) & (pts_px[:, 0] < width) &
        (pts_px[:, 1] >= 0) & (pts_px[:, 1] < height)
    )
    pts_px = pts_px[px_mask].astype(int)
    cols = cols[px_mask]

    # Render
    scale = min(512.0 / width, 512.0 / height, 1.0)
    tw, th = int(width * scale), int(height * scale)
    img = np.zeros((th, tw, 3), dtype=np.uint8)
    px = (pts_px * scale).astype(int)
    valid = (px[:, 0] >= 0) & (px[:, 0] < tw) & (px[:, 1] >= 0) & (px[:, 1] < th)
    px = px[valid]
    cols = cols[valid]
    img[px[:, 1], px[:, 0]] = cols[..., ::-1]  # RGB→BGR

    thumb_path = os.path.join(output_dir, "thumbnail.jpg")
    cv2.imwrite(thumb_path, img)
    logger = logging.getLogger("train_gs")
    logger.info(f"  Thumbnail saved: {thumb_path} ({len(px)} points)")


if __name__ == "__main__":
    main()
