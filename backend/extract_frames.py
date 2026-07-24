"""
extract_frames.py — Extract JPEG frames from video files for COLMAP.

Given a .MOV video path, this module:
  1. Opens the video with OpenCV (cv2)
  2. Extracts frames every N frames (configurable)
  3. Resizes to ~1008px wide (configurable)
  4. Saves as JPEG to frames/<video_name>/

Usage:
    python extract_frames.py /c/Users/Tolch/.../videos/IMG_0058.MOV
    python extract_frames.py /c/Users/Tolch/.../videos/IMG_0058.MOV --every 10 --max-frames 300
    python extract_frames.py /c/Users/Tolch/.../videos/IMG_0058.MOV --every 15 --max-frames 200 --width 1008
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths — project root detection
# ---------------------------------------------------------------------------

def find_project_root(start: str = None) -> str:
    """Walk up from start (or cwd) until we find config/settings.yaml."""
    if start is None:
        start = os.path.dirname(os.path.abspath(__file__))
    p = Path(start).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "config" / "settings.yaml").exists():
            return parent.as_posix()
    # fallback
    return p.as_posix()


PROJECT_ROOT = find_project_root()
LOGS_DIR = f"{PROJECT_ROOT}/logs"
FRAMES_DIR = f"{PROJECT_ROOT}/frames"

# Try to load config
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
# Status / progress helpers
# ---------------------------------------------------------------------------

def update_status(video_name: str, step: str, progress: int, message: str,
                  error: Optional[str] = None):
    """Write incremental status JSON so the webapp can poll it."""
    os.makedirs(f"{LOGS_DIR}/{video_name}", exist_ok=True)
    status = {
        "step": step,
        "progress": progress,
        "message": message,
    }
    if error:
        status["error"] = error
    status_path = f"{LOGS_DIR}/{video_name}/status.json"
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)


def setup_logging(video_name: str) -> logging.Logger:
    """Configure file + console logging for this video."""
    os.makedirs(f"{LOGS_DIR}/{video_name}", exist_ok=True)
    log_path = f"{LOGS_DIR}/{video_name}/pipeline.log"

    logger = logging.getLogger(f"extract_frames_{video_name}")
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


# ---------------------------------------------------------------------------
# Core frame extraction (OpenCV)
# ---------------------------------------------------------------------------

def extract_frames(video_path: str,
                   output_dir: Optional[str] = None,
                   every_n: Optional[int] = None,
                   max_frames: Optional[int] = None,
                   target_width: Optional[int] = None,
                   quality: Optional[int] = None,
                   progress_callback=None) -> str:
    """
    Extract JPEG frames from a video.

    Args:
        video_path: Absolute path to .MOV (or any video file).
        output_dir: Override output directory.  If None, auto-derived.
        every_n: Extract every Nth frame.  Default from config or 15.
        max_frames: Cap on total frames.  Default from config or 200.
        target_width: Resize width.  Default from config or 1008.
        quality: JPEG quality 1-100.  Default from config or 85.
        progress_callback: Optional callable(status_dict).

    Returns:
        Path to the frames output directory (MSYS2 style).
    """
    # ----- defaults from config -------------------------------------------
    cfg = _config if isinstance(_config, dict) else {}
    every_n = every_n if every_n is not None else cfg.get("extract_every_n", 15)
    max_frames = max_frames if max_frames is not None else cfg.get("max_frames", 200)
    target_width = target_width if target_width is not None else cfg.get("frame_width", 1008)
    quality = quality if quality is not None else cfg.get("frame_quality", 85)

    # ----- resolve paths --------------------------------------------------
    vp = Path(video_path.replace("\\", "/"))
    video_name = vp.stem  # e.g. "IMG_0058"

    if output_dir is None:
        output_dir = f"{FRAMES_DIR}/{video_name}"
    out = Path(output_dir.replace("\\", "/"))
    out.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(video_name)
    logger.info("=" * 60)
    logger.info("Frame extraction STARTED")
    logger.info(f"  Video:   {video_path}")
    logger.info(f"  Output:  {output_dir}")
    logger.info(f"  Every:   {every_n} frames")
    logger.info(f"  Max:     {max_frames} frames")
    logger.info(f"  Width:   {target_width} px")
    logger.info(f"  Quality: {quality}")

    if progress_callback:
        progress_callback({
            "step": "extract_frames",
            "progress": 0,
            "message": f"Opening video {video_name}...",
        })
    update_status(video_name, "extract_frames", 0, "Opening video...")

    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) not installed.  Run: pip install opencv-python-headless")
        raise

    # ----- open video -----------------------------------------------------
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        msg = f"Could not open video: {video_path}"
        logger.error(msg)
        update_status(video_name, "extract_frames", 0, msg, error=msg)
        raise RuntimeError(msg)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"  Frames:  {total_frames} total @ {fps:.2f} fps")

    if total_frames <= 0:
        cap.release()
        msg = f"Video has zero frames?  total_frames={total_frames}"
        logger.error(msg)
        update_status(video_name, "extract_frames", 0, msg, error=msg)
        raise RuntimeError(msg)

    # ----- extraction loop ------------------------------------------------
    frame_idx = 0
    saved = 0
    errors = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_idx % every_n == 0:
            try:
                h, w = frame.shape[:2]
                # Resize to target_width, keeping aspect ratio
                if w != target_width:
                    scale = target_width / w
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (target_width, new_h),
                                       interpolation=cv2.INTER_AREA)

                out_path = str(out / f"frame_{saved:06d}.jpg")
                success = cv2.imwrite(out_path, frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not success:
                    logger.warning(f"  Failed to write {out_path}")
                    errors += 1
                else:
                    saved += 1

                # Progress update every 10 saved frames
                if saved % 10 == 0 or saved == max_frames:
                    pct = min(int(saved / max_frames * 100), 99)
                    msg = f"Extracted {saved}/{max_frames} frames (frame #{frame_idx})"
                    logger.info(f"  {msg}")
                    update_status(video_name, "extract_frames", pct, msg)
                    if progress_callback:
                        progress_callback({
                            "step": "extract_frames",
                            "progress": pct,
                            "message": msg,
                        })

            except Exception as e:
                logger.warning(f"  Error processing frame {frame_idx}: {e}")
                errors += 1

        frame_idx += 1

    cap.release()
    logger.info(f"  Done — saved {saved} frames, skipped {frame_idx - saved}, errors {errors}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("Frame extraction COMPLETED")

    update_status(video_name, "extract_frames", 100,
                  f"Extracted {saved} frames")
    if progress_callback:
        progress_callback({
            "step": "extract_frames",
            "progress": 100,
            "message": f"Extracted {saved} frames from {video_name}",
        })

    return output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video for COLMAP")
    parser.add_argument("video_path", help="Path to .MOV video file")
    parser.add_argument("--every", type=int, default=None,
                        help=f"Extract every Nth frame (default: {_config.get('extract_every_n', 15)})")
    parser.add_argument("--max-frames", type=int, default=None,
                        help=f"Maximum frames to extract (default: {_config.get('max_frames', 200)})")
    parser.add_argument("--width", type=int, default=None,
                        help=f"Resize width (default: {_config.get('frame_width', 1008)})")
    parser.add_argument("--quality", type=int, default=None,
                        help=f"JPEG quality (default: {_config.get('frame_quality', 85)})")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")

    args = parser.parse_args()
    extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        every_n=args.every,
        max_frames=args.max_frames,
        target_width=args.width,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
