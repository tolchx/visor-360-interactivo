"""
FastAPI backend for the Teatro Colón Gaussian Splat Dashboard.

Serves the frontend, manages pipeline execution, and provides
REST endpoints for video status, previews, and GS output.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure the project root is on sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import (
    list_videos,
    read_status,
    start_pipeline,
    get_task_status,
    cancel_task,
    run_pipeline_for_server as run_pipeline,
    get_log_content,
    _get_ply_path,
    _get_thumbnail_path,
    VIDEOS_DIR,
    LOGS_DIR,
    GS_OUTPUT_DIR,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")

# --- Frontend path ---
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# --- Active WebSocket connections ---
active_connections: list[WebSocket] = []

# --- Task tracking ---
# Background task handles so we can reference them
_task_handles: dict[str, BackgroundTasks] = {}


def broadcast_status(video_name: str, status: str, progress: int, step: str):
    """Broadcast status update to all connected WebSocket clients."""
    payload = json.dumps({
        "type": "status_update",
        "video_name": video_name,
        "status": status,
        "progress": progress,
        "step": step,
        "timestamp": time.time(),
    })
    # Use asyncio to send to all connected clients
    import asyncio
    for ws in active_connections.copy():
        try:
            # We schedule the coroutine; if it can't be done, skip
            asyncio.create_task(ws.send_text(payload))
        except Exception:
            active_connections.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup/shutdown."""
    logger.info("Teatro Colón GS Dashboard starting up...")
    # Ensure directories exist
    for d in [LOGS_DIR, GS_OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Teatro Colón GS Dashboard shutting down...")


app = FastAPI(
    title="Teatro Colón Gaussian Splat Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/videos")
async def api_list_videos():
    """List all videos with their processing status."""
    videos = list_videos()
    return {"videos": videos, "count": len(videos)}


class PipelineStartRequest(BaseModel):
    video_name: str


@app.post("/api/pipeline/start")
async def api_start_pipeline(
    req: PipelineStartRequest,
    background_tasks: BackgroundTasks,
):
    """Start the pipeline for a given video."""
    video_name = req.video_name.strip()

    # Validate video exists
    video_path = VIDEOS_DIR / f"{video_name}.mov"
    if not video_path.exists():
        for ext in (".mp4", ".avi", ".mkv", ".webm"):
            alt = VIDEOS_DIR / f"{video_name}{ext}"
            if alt.exists():
                video_path = alt
                break
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Video '{video_name}' not found. Available: {[v['name'] for v in list_videos()]}",
            )

    # Check if already running
    status_data = read_status(video_name)
    if status_data.get("status") in ("processing",):
        raise HTTPException(
            status_code=409,
            detail=f"Video '{video_name}' is already being processed",
        )

    # Start pipeline
    task_id = start_pipeline(video_name)

    # Add background task
    background_tasks.add_task(
        run_pipeline,
        video_name,
        task_id,
        broadcast_status,
    )

    _task_handles[task_id] = background_tasks

    logger.info(f"Pipeline started: video={video_name}, task_id={task_id}")
    return {"task_id": task_id, "video_name": video_name, "status": "queued"}


@app.get("/api/pipeline/status/{task_id}")
async def api_pipeline_status(task_id: str):
    """Get the status of a pipeline task."""
    task = get_task_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


@app.get("/api/preview/{video_name}")
async def api_preview(video_name: str):
    """Serve a thumbnail/preview image for a video."""
    thumb_path = _get_thumbnail_path(video_name)
    if thumb_path.exists():
        return FileResponse(str(thumb_path), media_type="image/jpeg")

    # Fallback: serve a placeholder or video file's first frame
    # For now return 404
    raise HTTPException(status_code=404, detail=f"No preview for '{video_name}'")


@app.get("/api/gs/{video_name}")
async def api_gs(video_name: str):
    """Serve the output .ply file for a completed pipeline."""
    ply_path = _get_ply_path(video_name)
    if ply_path.exists():
        return FileResponse(
            str(ply_path),
            media_type="application/octet-stream",
            filename=f"{video_name}.ply",
        )
    raise HTTPException(
        status_code=404,
        detail=f"No GS output found for '{video_name}'. Path checked: {ply_path}",
    )


@app.get("/api/gs/{video_name}/meta")
async def api_gs_meta(video_name: str):
    """Get metadata about the GS output file."""
    ply_path = _get_ply_path(video_name)
    if ply_path.exists():
        stats = ply_path.stat()
        return {
            "video_name": video_name,
            "exists": True,
            "size": stats.st_size,
            "size_formatted": _format_size(stats.st_size),
            "modified": stats.st_mtime,
            "path": str(ply_path),
        }
    return {
        "video_name": video_name,
        "exists": False,
        "size": 0,
        "size_formatted": "0 B",
    }


@app.get("/api/logs/{video_name}")
async def api_logs(video_name: str):
    """Return the pipeline log content for a video."""
    content = get_log_content(video_name)
    return {"video_name": video_name, "log": content}


@app.delete("/api/pipeline/cancel/{task_id}")
async def api_cancel_pipeline(task_id: str):
    """Cancel a running pipeline task."""
    success = cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {"task_id": task_id, "status": "cancelled"}


# =============================================================================
# WebSocket endpoint for real-time status
# =============================================================================

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for real-time status updates."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket client connected ({len(active_connections)} total)")

    try:
        while True:
            # Wait for any message (keepalive or command)
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif msg.get("type") == "get_videos":
                    videos = list_videos()
                    await websocket.send_text(json.dumps({
                        "type": "videos_list",
                        "videos": videos,
                    }))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected ({len(active_connections)} remaining)")


# =============================================================================
# Static frontend serving
# =============================================================================

# Serve the main index.html at the root
@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found</h1><p>Build the frontend first.</p>")


# Serve static files from frontend/
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# =============================================================================
# Health check
# =============================================================================

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Teatro Colón GS Dashboard", "version": "1.0.0"}


# =============================================================================
# Utilities
# =============================================================================

def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Teatro Colón GS Dashboard on http://0.0.0.0:8765")
    print(f"Frontend: {FRONTEND_DIR}")
    print(f"Videos: {VIDEOS_DIR}")
    print(f"Logs: {LOGS_DIR}")
    print(f"GS Output: {GS_OUTPUT_DIR}")
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        log_level="info",
    )
