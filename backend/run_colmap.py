"""
run_colmap.py — Run COLMAP SfM pipeline on extracted frames.

Given a frames folder (from extract_frames.py), this module:
  1. Runs COLMAP feature_extractor (SIFT on GPU)
  2. Runs COLMAP sequential_matcher (fast matching for video sequences)
  3. Runs COLMAP mapper (sparse reconstruction)
  4. Saves outputs to colmap_input/<video_name>/

Usage:
    python run_colmap.py /c/Users/Tolch/.../frames/IMG_0058
    python run_colmap.py /c/Users/Tolch/.../frames/IMG_0058 --matcher sequential
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

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
COLMAP_INPUT_DIR = f"{PROJECT_ROOT}/colmap_input"
COLMAP_EXE = f"{PROJECT_ROOT}/colmap-bin/bin/colmap.exe"

# Config
_config = {}
_config_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
try:
    import yaml
    if os.path.exists(_config_path):
        with open(_config_path) as f:
            _config = yaml.safe_load(f) or {}
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    logger = logging.getLogger(f"run_colmap_{video_name}")
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


def to_msys2(path: str) -> str:
    """Convert any pathlike to MSYS2-style forward-slash for terminal output."""
    return path.replace("\\", "/")


# ---------------------------------------------------------------------------
# COLMAP runner
# ---------------------------------------------------------------------------

def run_colmap(subcommand: str, args_list: list, logger: logging.Logger,
               cwd: str = None) -> int:
    """Run a colmap subcommand via subprocess and return the returncode."""
    cmd = [COLMAP_EXE, subcommand] + args_list
    cmd_str = " ".join(str(x) for x in cmd)
    logger.info(f"  Running: {cmd_str}")
    logger.debug(f"  CWD: {cwd or 'inherited'}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1h per step
        )
    except subprocess.TimeoutExpired:
        msg = f"COLMAP {subcommand} timed out after 3600s"
        logger.error(msg)
        raise RuntimeError(msg)
    except FileNotFoundError:
        msg = f"COLMAP executable not found at: {COLMAP_EXE}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Log output
    for line in result.stdout.splitlines():
        logger.debug(f"  [colmap stdout] {line}")
    for line in result.stderr.splitlines():
        logger.warning(f"  [colmap stderr] {line}")

    if result.returncode != 0:
        logger.error(f"COLMAP {subcommand} failed with code {result.returncode}")
        # Collect last 10 lines of stderr for the error message
        err_lines = result.stderr.strip().splitlines()[-10:]
        raise RuntimeError(
            f"COLMAP {subcommand} failed (code {result.returncode}): "
            f"{'; '.join(err_lines)}"
        )

    logger.info(f"  COLMAP {subcommand} completed (code {result.returncode})")
    return result.returncode


def check_colmap_exists():
    """Verify COLMAP binary exists, raise if not."""
    exe = Path(COLMAP_EXE)
    if not exe.exists():
        raise FileNotFoundError(
            f"COLMAP executable not found at: {COLMAP_EXE}. "
            f"Please check config/settings.yaml or install COLMAP."
        )
    return str(exe)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_sfm(frames_dir: str,
            output_dir: Optional[str] = None,
            matcher: Optional[str] = None,
            sift_gpu_index: Optional[int] = None,
            progress_callback=None) -> str:
    """
    Run COLMAP SfM on extracted frames.

    Args:
        frames_dir: Path to frames directory (e.g., frames/IMG_0058).
        output_dir: Override output dir for colmap results. Auto-derived if None.
        matcher: 'sequential', 'exhaustive', or 'vocab_tree'. Config default.
        sift_gpu_index: GPU index for SIFT (default: 0).
        progress_callback: Optional callable(status_dict).

    Returns:
        Path to the sparse/0 COLMAP output directory.
    """
    check_colmap_exists()

    # Config defaults
    colmap_cfg = _config.get("colmap", {}) if isinstance(_config, dict) else {}
    matcher = matcher or colmap_cfg.get("matcher", "sequential")
    sift_gpu_index = sift_gpu_index if sift_gpu_index is not None else colmap_cfg.get("sift_gpu_index", 0)

    # Derive video name from frames directory
    frames_path = Path(frames_dir.replace("\\", "/"))
    video_name = frames_path.name  # e.g. "IMG_0058"

    if output_dir is None:
        output_dir = f"{COLMAP_INPUT_DIR}/{video_name}"
    sparse_dir = f"{output_dir}/sparse/0"
    db_path = f"{output_dir}/database.db"

    logger = setup_logging(video_name)
    logger.info("=" * 60)
    logger.info("COLMAP SfM STARTED")
    logger.info(f"  Frames:  {frames_dir}")
    logger.info(f"  Output:  {output_dir}")
    logger.info(f"  DB:      {db_path}")
    logger.info(f"  Matcher: {matcher}")
    logger.info(f"  GPU:     {sift_gpu_index}")

    # Create output dirs
    Path(output_dir.replace("\\", "/")).mkdir(parents=True, exist_ok=True)
    Path(sparse_dir.replace("\\", "/")).mkdir(parents=True, exist_ok=True)

    # Remove existing database to start fresh
    db = Path(db_path.replace("\\", "/"))
    if db.exists():
        db.unlink()
        logger.info("  Removed existing database")

    update_status(video_name, "run_colmap", 0, "Starting COLMAP...")
    if progress_callback:
        progress_callback({
            "step": "run_colmap",
            "progress": 0,
            "message": f"Starting COLMAP SfM for {video_name}...",
        })

    # -----------------------------------------------------------------------
    # Step 1: Feature extraction
    # -----------------------------------------------------------------------
    logger.info("--- Step 1/3: Feature extraction ---")
    update_status(video_name, "run_colmap", 5, "Extracting SIFT features...")
    if progress_callback:
        progress_callback({
            "step": "run_colmap",
            "progress": 5,
            "message": "Extracting SIFT features with COLMAP...",
        })

    run_colmap("feature_extractor", [
        "--database_path", db_path,
        "--image_path", frames_dir,
        "--FeatureExtraction.use_gpu", "1",
    ], logger)

    logger.info("  Feature extraction complete")
    update_status(video_name, "run_colmap", 30, "Feature extraction complete")
    if progress_callback:
        progress_callback({
            "step": "run_colmap",
            "progress": 30,
            "message": "SIFT features extracted",
        })

    # -----------------------------------------------------------------------
    # Step 2: Feature matching
    # -----------------------------------------------------------------------
    logger.info(f"--- Step 2/3: Feature matching ({matcher}) ---")
    update_status(video_name, "run_colmap", 35, f"Matching features ({matcher})...")
    if progress_callback:
        progress_callback({
            "step": "run_colmap",
            "progress": 35,
            "message": f"Matching features using {matcher} matcher...",
        })

    match_args = ["--database_path", db_path, "--FeatureMatching.use_gpu", "1"]

    if matcher == "sequential":
        match_args.extend(["--SequentialMatching.overlap", "10",
                           "--SequentialMatching.loop_detection", "0"])
        run_colmap("sequential_matcher", match_args, logger)
    elif matcher == "exhaustive":
        run_colmap("exhaustive_matcher", match_args, logger)
    elif matcher == "vocab_tree":
        vocab_path = colmap_cfg.get("vocab_tree")
        if vocab_path and os.path.exists(vocab_path):
            match_args.extend(["--VocabTreeMatching.vocab_tree_path", vocab_path])
        run_colmap("vocab_tree_matcher", match_args, logger)
    else:
        logger.warning(f"  Unknown matcher '{matcher}', falling back to sequential")
        match_args.extend(["--SequentialMatching.overlap", "10"])
        run_colmap("sequential_matcher", match_args, logger)

    logger.info("  Feature matching complete")
    update_status(video_name, "run_colmap", 60, "Feature matching complete")
    if progress_callback:
        progress_callback({
            "step": "run_colmap",
            "progress": 60,
            "message": "Feature matching complete",
        })

    # -----------------------------------------------------------------------
    # Step 3: Sparse reconstruction (mapper)
    # -----------------------------------------------------------------------
    logger.info("--- Step 3/3: Sparse reconstruction (mapper) ---")
    update_status(video_name, "run_colmap", 65, "Running sparse reconstruction...")
    if progress_callback:
        progress_callback({
            "step": "run_colmap",
            "progress": 65,
            "message": "Running COLMAP mapper (sparse reconstruction)...",
        })

    run_colmap("mapper", [
        "--database_path", db_path,
        "--image_path", frames_dir,
        "--output_path", f"{output_dir}/sparse",
    ], logger)

    # Check if sparse reconstruction produced valid output
    sparse_path = Path(sparse_dir.replace("\\", "/"))
    cameras_bin = sparse_path / "cameras.bin"
    images_bin = sparse_path / "images.bin"
    points3d_bin = sparse_path / "points3D.bin"

    if not (cameras_bin.exists() and images_bin.exists() and points3d_bin.exists()):
        # Try alternative — check if output went to sparse/0/ or similar
        logger.warning("  Expected files not in sparse/0/, checking alternatives...")
        # List what's in the sparse directory
        parent_sparse = Path(f"{output_dir}/sparse".replace("\\", "/"))
        if parent_sparse.exists():
            subdirs = sorted(parent_sparse.iterdir())
            logger.info(f"  Contents of sparse/: {[d.name for d in subdirs]}")
            # Try to find a subfolder with the right files
            for sub in subdirs:
                if sub.is_dir() and (sub / "cameras.bin").exists():
                    sparse_dir = str(sub)
                    sparse_path = sub
                    cameras_bin = sub / "cameras.bin"
                    images_bin = sub / "images.bin"
                    points3d_bin = sub / "points3D.bin"
                    logger.info(f"  Found output in: {sparse_dir}")
                    break

    if not (cameras_bin.exists() and images_bin.exists() and points3d_bin.exists()):
        raise RuntimeError(
            f"COLMAP mapper did not produce expected output in {sparse_dir}. "
            f"cameras.bin exists: {cameras_bin.exists()}, "
            f"images.bin exists: {images_bin.exists()}, "
            f"points3D.bin exists: {points3d_bin.exists()}"
        )

    logger.info("  Sparse reconstruction complete")
    logger.info(f"  Cameras:  {cameras_bin}")
    logger.info(f"  Images:   {images_bin}")
    logger.info(f"  Points3D: {points3d_bin}")

    update_status(video_name, "run_colmap", 100,
                  f"COLMAP SfM complete — output in {sparse_dir}")
    if progress_callback:
        progress_callback({
            "step": "run_colmap",
            "progress": 100,
            "message": f"COLMAP SfM complete — {sparse_dir}",
        })

    return str(sparse_path.as_posix())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP SfM on extracted frames")
    parser.add_argument("frames_dir", help="Path to frames directory")
    parser.add_argument("--output-dir", default=None,
                        help="Override COLMAP output directory")
    parser.add_argument("--matcher", default=None,
                        choices=["sequential", "exhaustive", "vocab_tree"],
                        help="Feature matcher type")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="GPU index for SIFT extraction")

    args = parser.parse_args()
    result = run_sfm(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        matcher=args.matcher,
        sift_gpu_index=args.gpu_index,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
