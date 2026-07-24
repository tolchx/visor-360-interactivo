"""
pipeline.py — Orchestrator for the Teatro Colón Gaussian Splat pipeline.

This module ties together:
  1. extract_frames.extract_frames()
  2. run_colmap.run_sfm()
  3. train_gs.train_gs()

It supports running the full pipeline or individual steps, reports progress
via callbacks, handles errors gracefully, and writes status JSON for the webapp.

Usage:
    from pipeline import run_pipeline
    run_pipeline("/c/Users/Tolch/.../videos/IMG_0058.MOV",
                 callback=lambda s: print(s))

    # Run specific steps only
    run_pipeline(video, steps=["extract"], callback=...)
    run_pipeline(video, steps=["colmap"], callback=...)
    run_pipeline(video, steps=["train"], callback=...)
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Optional

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
FRAMES_DIR = f"{PROJECT_ROOT}/frames"
COLMAP_INPUT_DIR = f"{PROJECT_ROOT}/colmap_input"
GS_OUTPUT_DIR = f"{PROJECT_ROOT}/gs_output"

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


# ===================================================================
# Pipeline result tracking
# ===================================================================

class PipelineResult:
    """Structured result from a pipeline run."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_name = Path(video_path.replace("\\", "/")).stem
        self.success = False
        self.error: Optional[str] = None
        self.steps_completed: List[str] = []
        self.frames_dir: Optional[str] = None
        self.colmap_dir: Optional[str] = None
        self.ply_path: Optional[str] = None
        self.thumbnail_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "video_name": self.video_name,
            "success": self.success,
            "error": self.error,
            "steps_completed": self.steps_completed,
            "frames_dir": self.frames_dir,
            "colmap_dir": self.colmap_dir,
            "ply_path": self.ply_path,
            "thumbnail_path": self.thumbnail_path,
        }


# ===================================================================
# Status helpers
# ===================================================================

def update_status(video_name: str, step: str, progress: int, message: str,
                  error: Optional[str] = None):
    """Write incremental status JSON so the webapp can poll it."""
    os.makedirs(f"{LOGS_DIR}/{video_name}", exist_ok=True)
    # Determine status from progress/error
    if error:
        status_val = "error"
    elif progress >= 100:
        status_val = "done"
    elif progress > 0:
        status_val = "processing"
    else:
        status_val = "queued"
    status = {"step": step, "progress": progress, "message": message,
              "status": status_val}
    if error:
        status["error"] = error
    with open(f"{LOGS_DIR}/{video_name}/status.json", "w") as f:
        json.dump(status, f, indent=2)


def setup_logging(video_name: str) -> logging.Logger:
    os.makedirs(f"{LOGS_DIR}/{video_name}", exist_ok=True)
    log_path = f"{LOGS_DIR}/{video_name}/pipeline.log"
    logger = logging.getLogger(f"pipeline_{video_name}")
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
# Pipeline runner
# ===================================================================

def run_pipeline(
    video_path: str,
    steps: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
    skip_extract: bool = False,
    skip_colmap: bool = False,
    skip_train: bool = False,
    extract_every_n: Optional[int] = None,
    extract_max_frames: Optional[int] = None,
    colmap_matcher: Optional[str] = None,
    train_iterations: Optional[int] = None,
    train_lr: Optional[float] = None,
) -> PipelineResult:
    """
    Run the full Gaussian Splat conversion pipeline for a single video.

    Args:
        video_path: Absolute path to a .MOV video file.
        steps: List of steps to run.  Default: ['extract', 'colmap', 'train'].
               Set to a subset to run only specific steps.
        callback: Callable(status_dict) called on each progress update.
        skip_extract: If True, skip frame extraction (use existing frames).
        skip_colmap: If True, skip COLMAP (use existing reconstruction).
        skip_train: If True, skip training (just prepare data).
        extract_every_n: Override frame extraction interval.
        extract_max_frames: Override max frames to extract.
        colmap_matcher: Override COLMAP matcher type.
        train_iterations: Override training iterations.
        train_lr: Override training learning rate.

    Returns:
        PipelineResult with success/failure info.
    """
    # Clean path
    video_path = video_path.replace("\\", "/")
    video_name = Path(video_path).stem

    result = PipelineResult(video_path)
    logger = setup_logging(video_name)

    logger.info("=" * 70)
    logger.info("  TEATRO COLÓN — Gaussian Splat Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Video:  {video_path}")
    logger.info(f"  Name:   {video_name}")

    # Determine which steps to run
    if steps is not None:
        run_steps = steps
    else:
        run_steps = []
        if not skip_extract:
            run_steps.append("extract")
        if not skip_colmap:
            run_steps.append("colmap")
        if not skip_train:
            run_steps.append("train")

    logger.info(f"  Steps:  {', '.join(run_steps)}")

    # ---- Step 0: Validate video -------------------------------------------
    if not os.path.exists(video_path):
        msg = f"Video file not found: {video_path}"
        logger.error(msg)
        result.error = msg
        update_status(video_name, "pipeline", 0, msg, error=msg)
        if callback:
            callback({"step": "pipeline", "progress": 0, "message": msg, "error": msg})
        return result

    vp = Path(video_path)
    if vp.suffix.lower() not in (".mov", ".mp4", ".avi", ".m4v", ".webm"):
        logger.warning(f"  Unexpected video extension: {vp.suffix}")

    update_status(video_name, "pipeline", 0, f"Starting pipeline for {video_name}")
    if callback:
        callback({"step": "pipeline", "progress": 0, "message": f"Starting pipeline for {video_name}"})

    try:
        # ---- Step 1: Frame Extraction ------------------------------------
        if "extract" in run_steps:
            logger.info("")
            logger.info("--- Step 1/3: Frame Extraction ---")
            update_status(video_name, "pipeline", 5, "Extracting frames...")
            if callback:
                callback({"step": "pipeline", "progress": 5,
                          "message": "Extracting frames from video..."})

            from backend.extract_frames import extract_frames

            frames_dir = extract_frames(
                video_path=video_path,
                every_n=extract_every_n,
                max_frames=extract_max_frames,
                progress_callback=callback,
            )
            result.frames_dir = frames_dir
            result.steps_completed.append("extract")
            logger.info(f"  Frames directory: {frames_dir}")
        else:
            # Use existing frames dir
            result.frames_dir = f"{FRAMES_DIR}/{video_name}"
            if not os.path.exists(result.frames_dir) or \
               len(os.listdir(result.frames_dir)) == 0:
                logger.warning(f"  No existing frames found at {result.frames_dir}")
            else:
                n_frames = len([f for f in os.listdir(result.frames_dir)
                                if f.endswith(".jpg")])
                logger.info(f"  Using existing frames ({n_frames} files)")

        # ---- Step 2: COLMAP SfM -----------------------------------------
        if "colmap" in run_steps:
            logger.info("")
            logger.info("--- Step 2/3: COLMAP SfM ---")
            update_status(video_name, "pipeline", 35, "Running COLMAP SfM...")
            if callback:
                callback({"step": "pipeline", "progress": 35,
                          "message": "Running COLMAP Structure-from-Motion..."})

            from backend.run_colmap import run_sfm

            sparse_dir = run_sfm(
                frames_dir=result.frames_dir,
                matcher=colmap_matcher,
                progress_callback=callback,
            )
            result.colmap_dir = sparse_dir
            result.steps_completed.append("colmap")
            logger.info(f"  COLMAP output: {sparse_dir}")
        else:
            # Use existing COLMAP output
            candidate = f"{COLMAP_INPUT_DIR}/{video_name}/sparse/0"
            if os.path.exists(f"{candidate}/cameras.bin"):
                result.colmap_dir = candidate
                logger.info(f"  Using existing COLMAP: {candidate}")
            else:
                candidate2 = f"{COLMAP_INPUT_DIR}/{video_name}/sparse"
                if os.path.exists(candidate2):
                    # Check subdirs
                    subdirs = sorted(Path(candidate2).iterdir())
                    for sub in subdirs:
                        if sub.is_dir() and (sub / "cameras.bin").exists():
                            result.colmap_dir = str(sub)
                            logger.info(f"  Using existing COLMAP: {result.colmap_dir}")
                            break
                if result.colmap_dir is None:
                    logger.warning(f"  No existing COLMAP output found")

        # ---- Step 3: GS Training -----------------------------------------
        if "train" in run_steps and result.colmap_dir:
            logger.info("")
            logger.info("--- Step 3/3: Gaussian Splat Training ---")
            update_status(video_name, "pipeline", 65, "Training Gaussian Splats...")
            if callback:
                callback({"step": "pipeline", "progress": 65,
                          "message": "Training 3D Gaussian Splats..."})

            from backend.train_gs import train_gs

            ply_path = train_gs(
                colmap_sparse_dir=result.colmap_dir,
                frames_dir=result.frames_dir,
                num_iterations=train_iterations,
                learning_rate=train_lr,
                progress_callback=callback,
            )
            result.ply_path = ply_path
            result.steps_completed.append("train")
            logger.info(f"  GS model: {ply_path}")

            # Check for thumbnail
            thumb_path = f"{GS_OUTPUT_DIR}/{video_name}/thumbnail.jpg"
            if os.path.exists(thumb_path):
                result.thumbnail_path = thumb_path
                logger.info(f"  Thumbnail: {thumb_path}")

        elif "train" in run_steps and not result.colmap_dir:
            msg = "Cannot run training: no COLMAP output available"
            logger.error(msg)
            raise RuntimeError(msg)

        # ---- Success ------------------------------------------------------
        result.success = True
        logger.info("")
        logger.info("=" * 70)
        logger.info("  PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"  Frames:  {result.frames_dir}")
        logger.info(f"  COLMAP:  {result.colmap_dir}")
        logger.info(f"  GS:      {result.ply_path}")
        logger.info("=" * 70)

        update_status(video_name, "pipeline", 100,
                      f"Pipeline complete for {video_name}")
        if callback:
            callback({
                "step": "pipeline",
                "progress": 100,
                "message": f"Pipeline complete for {video_name}",
            })

    except Exception as e:
        logger.error(f"  Pipeline FAILED: {e}")
        logger.error(traceback.format_exc())
        result.success = False
        result.error = str(e)
        update_status(video_name, "pipeline", -1, f"Pipeline failed: {e}", error=str(e))
        if callback:
            callback({
                "step": "pipeline",
                "progress": -1,
                "message": f"Pipeline failed: {e}",
                "error": str(e),
            })

    # Write final result JSON
    result_path = f"{PROJECT_ROOT}/results/{video_name}_result.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"  Result saved to {result_path}")

    return result


# ===================================================================
# Batch processing
# ===================================================================

def run_batch(video_dir: str = None, pattern: str = "*.MOV",
              callback: Optional[Callable] = None, **kwargs) -> List[PipelineResult]:
    """
    Run the pipeline on all videos matching a pattern.

    Args:
        video_dir: Directory containing videos.  Default: PROJECT_ROOT/videos/.
        pattern: Glob pattern for video files (default: *.MOV).
        callback: Passed through to run_pipeline for each video.
        **kwargs: Additional args forwarded to run_pipeline.

    Returns:
        List of PipelineResult objects.
    """
    if video_dir is None:
        video_dir = f"{PROJECT_ROOT}/videos"
    video_path = Path(video_dir.replace("\\", "/"))

    videos = sorted(video_path.glob(pattern))
    if not videos:
        # Also try lowercase
        videos = sorted(video_path.glob(pattern.lower()))
    if not videos:
        print(f"No videos found in {video_dir} matching {pattern}")
        return []

    results = []
    for v in videos:
        print(f"\n{'=' * 70}")
        print(f"Processing: {v.name}")
        print(f"{'=' * 70}")
        result = run_pipeline(str(v), callback=callback, **kwargs)
        results.append(result)
        if not result.success:
            print(f"  FAILED: {result.error}")
        else:
            print(f"  OK — steps: {result.steps_completed}")

    # Summary
    successes = sum(1 for r in results if r.success)
    print(f"\nBatch complete: {successes}/{len(results)} succeeded")
    return results


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Teatro Colón Gaussian Splat Pipeline")
    parser.add_argument("video_path", nargs="?",
                        help="Path to a .MOV video file")
    parser.add_argument("--batch", action="store_true",
                        help="Process all .MOV files in the videos/ directory")
    parser.add_argument("--steps", nargs="+",
                        choices=["extract", "colmap", "train"],
                        help="Steps to run (default: all)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip frame extraction")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="Skip COLMAP")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training")
    parser.add_argument("--extract-every", type=int, default=None,
                        help="Extract every Nth frame")
    parser.add_argument("--extract-max-frames", type=int, default=None,
                        help="Max frames to extract")
    parser.add_argument("--colmap-matcher", default=None,
                        choices=["sequential", "exhaustive", "vocab_tree"],
                        help="COLMAP matcher type")
    parser.add_argument("--train-iter", type=int, default=None,
                        help="Training iterations")
    parser.add_argument("--train-lr", type=float, default=None,
                        help="Training learning rate")

    args = parser.parse_args()

    if args.batch:
        results = run_batch(
            steps=args.steps,
            skip_extract=args.skip_extract,
            skip_colmap=args.skip_colmap,
            skip_train=args.skip_train,
            extract_every_n=args.extract_every,
            extract_max_frames=args.extract_max_frames,
            colmap_matcher=args.colmap_matcher,
            train_iterations=args.train_iter,
            train_lr=args.train_lr,
        )
    elif args.video_path:
        result = run_pipeline(
            video_path=args.video_path,
            steps=args.steps,
            skip_extract=args.skip_extract,
            skip_colmap=args.skip_colmap,
            skip_train=args.skip_train,
            extract_every_n=args.extract_every,
            extract_max_frames=args.extract_max_frames,
            colmap_matcher=args.colmap_matcher,
            train_iterations=args.train_iter,
            train_lr=args.train_lr,
        )
        status = "SUCCESS" if result.success else "FAILED"
        print(f"\nPipeline {status}: {result.video_name}")
        if result.error:
            print(f"  Error: {result.error}")
    else:
        parser.print_help()


# ===================================================================
# Webapp bridge functions (used by server.py)
# ===================================================================

from pathlib import Path as _Path

# Directory constants for the webapp
VIDEOS_DIR = _Path(PROJECT_ROOT) / "videos"
LOGS_DIR = _Path(LOGS_DIR)  # override string LOGS_DIR with Path for server.py
GS_OUTPUT_DIR = _Path(GS_OUTPUT_DIR)  # override string GS_OUTPUT_DIR with Path for server.py

# In-memory task tracking
_tasks: dict = {}
_task_counter: int = 0


def list_videos() -> list:
    """List all videos with their processing status."""
    videos = []
    video_dir = _Path(PROJECT_ROOT) / "videos"
    if not video_dir.exists():
        return videos

    for f in sorted(video_dir.iterdir()):
        if f.suffix.lower() in (".mov", ".mp4", ".avi", ".mkv", ".webm"):
            name = f.stem
            status_data = read_status(name)
            gs_path = _get_ply_path(name)
            colmap_dir = f"{COLMAP_INPUT_DIR}/{name}/sparse/0"

            videos.append({
                "name": name,
                "path": str(f),
                "extension": f.suffix,
                "size": f.stat().st_size,
                "size_formatted": _format_size(f.stat().st_size),
                "status": status_data.get("status", "pending"),
                "progress": status_data.get("progress", 0),
                "step": status_data.get("step", ""),
                "message": status_data.get("message", ""),
                "error": status_data.get("error", ""),
                "has_frames": _Path(f"{FRAMES_DIR}/{name}").exists() and len(os.listdir(f"{FRAMES_DIR}/{name}")) > 0,
                "has_colmap": os.path.exists(f"{colmap_dir}/cameras.bin"),
                "has_gs": gs_path.exists(),
                "gs_size": gs_path.stat().st_size if gs_path.exists() else 0,
            })
    return videos


def read_status(video_name: str) -> dict:
    """Read the status JSON for a video."""
    status_path = f"{LOGS_DIR}/{video_name}/status.json"
    if os.path.exists(status_path):
        try:
            with open(status_path) as f:
                data = json.load(f)
            # Backward compat: derive status if missing
            if "status" not in data:
                err = data.get("error")
                prog = data.get("progress", 0)
                if err:
                    data["status"] = "error"
                elif prog >= 100:
                    data["status"] = "done"
                elif prog > 0:
                    data["status"] = "processing"
                else:
                    data["status"] = "pending"
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"status": "pending", "progress": 0, "step": "", "message": ""}


def _format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_ply_path(video_name: str) -> _Path:
    """Get the expected .ply path for a video."""
    return _Path(GS_OUTPUT_DIR) / video_name / "scene.ply"


def _get_thumbnail_path(video_name: str) -> _Path:
    """Get the expected thumbnail path for a video."""
    return _Path(GS_OUTPUT_DIR) / video_name / "thumbnail.jpg"


def get_log_content(video_name: str) -> str:
    """Return the pipeline log content for a video."""
    log_path = f"{LOGS_DIR}/{video_name}/pipeline.log"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return ""


# Task management for server.py background tasks

def start_pipeline(video_name: str) -> str:
    """Register a pipeline task and return its task_id."""
    global _task_counter
    _task_counter += 1
    task_id = f"task_{_task_counter:04d}"
    _tasks[task_id] = {
        "task_id": task_id,
        "video_name": video_name,
        "status": "queued",
        "progress": 0,
        "step": "",
        "message": "",
        "error": "",
    }
    update_status(video_name, "pipeline", 0, "Queued")
    return task_id


def get_task_status(task_id: str) -> Optional[dict]:
    """Get the status of a pipeline task."""
    return _tasks.get(task_id)


def cancel_task(task_id: str) -> bool:
    """Cancel a running pipeline task."""
    if task_id in _tasks:
        _tasks[task_id]["status"] = "cancelled"
        _tasks[task_id]["message"] = "Cancelled by user"
        return True
    return False


# Override the existing run_pipeline import so server.py can call it
# with (video_name, task_id, broadcast_callback) signature
def run_pipeline_for_server(video_name: str, task_id: str,
                             broadcast_callback=None):
    """Wrapper for server.py background tasks."""
    if task_id in _tasks:
        _tasks[task_id]["status"] = "processing"

    video_path = VIDEOS_DIR / f"{video_name}.mov"
    if not video_path.exists():
        for ext in (".mp4", ".avi", ".mkv", ".webm"):
            alt = VIDEOS_DIR / f"{video_name}{ext}"
            if alt.exists():
                video_path = alt
                break

    def progress_cb(status_dict):
        p = status_dict.get("progress", 0)
        msg = status_dict.get("message", "")
        step = status_dict.get("step", "")
        err = status_dict.get("error")

        if task_id in _tasks:
            if err:
                _tasks[task_id]["status"] = "error"
                _tasks[task_id]["error"] = err
            elif p >= 100:
                _tasks[task_id]["status"] = "completed"
            else:
                _tasks[task_id]["status"] = "processing"
            _tasks[task_id]["progress"] = p
            _tasks[task_id]["message"] = msg
            _tasks[task_id]["step"] = step

        if broadcast_callback:
            try:
                broadcast_callback(video_name, _tasks.get(task_id, {}).get("status", "processing"), p, step)
            except Exception:
                pass

    try:
        result = run_pipeline(str(video_path), callback=progress_cb)

        if task_id in _tasks:
            _tasks[task_id]["status"] = "completed" if result.success else "error"
            _tasks[task_id]["progress"] = 100 if result.success else -1
            if result.error:
                _tasks[task_id]["error"] = result.error
    except Exception as e:
        if task_id in _tasks:
            _tasks[task_id]["status"] = "error"
            _tasks[task_id]["error"] = str(e)


if __name__ == "__main__":
    main()
