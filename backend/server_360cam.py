"""
360Cam-PGM-3DGS-Tools Web Dashboard
FastAPI backend that wraps the 360° → perspective conversion tools.
"""
import json, os, sys, subprocess, time, threading, logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Paths ---
TOOLS_DIR = Path(r"C:\Users\Tolch\Documents\AI_Code\360Cam-PGM-3DGS-Tools")
CLI_DIR = TOOLS_DIR / "cli_tools"
PERSPCUT_SCRIPT = CLI_DIR / "gs360_360PerspCut.py"
VENV_PYTHON = r"C:\Users\Tolch\Documents\AI_Code\Teatro_Colon-GS\venv\Scripts\python.exe"

# 360 content source
CONTENT_DIR = Path(r"D:\Proyectos_Activos\2026-07-23 Le Parc Colon\360 contenido")

# Output
OUTPUT_DIR = Path(r"C:\Users\Tolch\Documents\AI_Code\Teatro_Colon-GS\Postshot")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gs360_web")

# --- Task tracking ---
_tasks = {}
_ws_clients = []

def broadcast(msg: dict):
    payload = json.dumps(msg)
    for ws in _ws_clients.copy():
        try:
            import asyncio
            asyncio.create_task(ws.send_text(payload))
        except: _ws_clients.remove(ws)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("360Cam Web Dashboard starting...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("360Cam Web Dashboard shutting down...")

app = FastAPI(title="360Cam → Postshot Dashboard", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ======= API =======

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "360Cam-PGM-3DGS Web"}

@app.get("/api/content")
async def list_content():
    """List all 360 photos and videos available for processing."""
    items = []
    # Scan content directory
    for root, dirs, files in os.walk(str(CONTENT_DIR)):
        for f in files:
            ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""
            if ext in ("jpg", "jpeg", "png", "tiff", "tif", "mp4", "mov", "insv", "lrv"):
                path = os.path.join(root, f)
                sz = os.path.getsize(path)
                is_video = ext in ("mp4", "mov", "insv", "lrv")
                items.append({
                    "name": f, "path": path, "size": sz,
                    "size_fmt": f"{sz/1e6:.1f}MB",
                    "type": "video" if is_video else "photo",
                    "source": os.path.relpath(path, str(CONTENT_DIR)),
                })
    return {"items": items, "count": len(items)}

@app.get("/api/presets")
async def get_presets():
    """Return available 360PerspCut presets."""
    return {"presets": [
        {"name": "RealityScan (default)", "count": 8, "fov": 90, "size": 1600, "focal_mm": 12},
        {"name": "Fisheye-like", "count": 6, "fov": 100, "size": 1600, "focal_mm": 17},
        {"name": "Full 360 coverage", "count": 12, "fov": 60, "size": 1200, "focal_mm": 14},
        {"name": "Quality (8K)", "count": 8, "fov": 90, "size": 3000, "focal_mm": 12},
        {"name": "Quick preview", "count": 4, "fov": 90, "size": 800, "focal_mm": 12},
    ]}

class PerspCutRequest(BaseModel):
    input_path: str
    preset: str = "RealityScan (default)"
    fps: float | None = None  # for video
    count: int | None = None
    size: int | None = None
    hfov: float | None = None

@app.post("/api/perspcut/start")
async def start_perspcut(req: PerspCutRequest, background_tasks: BackgroundTasks):
    """Start 360PerspCut processing."""
    task_id = f"perspcut_{int(time.time())}"
    _tasks[task_id] = {"status": "queued", "progress": 0, "message": "Queued..."}

    preset_map = {
        "RealityScan (default)": {"count": 8, "fov": 90, "size": 1600, "focal_mm": 12},
        "Fisheye-like": {"count": 6, "fov": 100, "size": 1600, "focal_mm": 17},
        "Full 360 coverage": {"count": 12, "fov": 60, "size": 1200, "focal_mm": 14},
        "Quality (8K)": {"count": 8, "fov": 90, "size": 3000, "focal_mm": 12},
        "Quick preview": {"count": 4, "fov": 90, "size": 800, "focal_mm": 12},
    }
    p = preset_map.get(req.preset, preset_map["RealityScan (default)"])
    count = req.count or p["count"]
    out_size = req.size or p["size"]
    hfov = req.hfov or p["fov"]
    focal_mm = p["focal_mm"]

    background_tasks.add_task(_run_perspcut, task_id, req.input_path, count, out_size, hfov, focal_mm, req.fps)
    return {"task_id": task_id}

def _run_perspcut(task_id, input_path, count, out_size, hfov, focal_mm, fps):
    _tasks[task_id] = {"status": "running", "progress": 5, "message": "Starting..."}
    broadcast({"type": "task_update", "task_id": task_id, "status": "running", "progress": 5})

    input_path = input_path.replace("\\", "/")
    # Derive output dir
    stem = Path(input_path).stem
    out_dir = str(OUTPUT_DIR / f"perspcut_{stem}")
    os.makedirs(out_dir, exist_ok=True)

    # Build command
    cmd = [
        str(VENV_PYTHON), str(PERSPCUT_SCRIPT),
        "-i", input_path,
        "-o", out_dir,
        "--count", str(count),
        "--size", str(out_size),
        "--hfov", str(hfov),
        "--focal-mm", str(focal_mm),
        "--jpeg-quality-95",
    ]
    if fps:
        cmd += ["-f", str(fps)]

    # For photos (not video), process all images in the input dir
    ext = input_path.lower().rsplit(".", 1)[-1] if "." in input_path else ""
    is_video = ext in ("mp4", "mov", "insv", "lrv")

    _tasks[task_id]["message"] = f"Running: {' '.join(str(c) for c in cmd[:8])}..."
    broadcast({"type": "task_update", "task_id": task_id, "status": "running", "progress": 10,
               "message": _tasks[task_id]["message"]})

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(CLI_DIR),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        # Monitor output
        last_msg = ""
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                logger.info(f"[{task_id}] {line}")
                last_msg = line
                # Estimate progress: look for percentage-like output
                for token in ("%", "frame", "output", "writing"):
                    if token in line.lower():
                        _tasks[task_id]["message"] = line[:120]
                        broadcast({"type": "task_update", "task_id": task_id, "status": "running",
                                   "message": line[:200]})
                        break
        proc.wait()
        if proc.returncode == 0:
            # Count output images
            img_count = len([f for f in os.listdir(out_dir) if f.endswith((".jpg", ".png"))])
            _tasks[task_id] = {"status": "done", "progress": 100,
                               "message": f"Done! {img_count} images → {out_dir}",
                               "output_dir": out_dir}
            broadcast({"type": "task_update", "task_id": task_id, "status": "done", "progress": 100,
                       "message": _tasks[task_id]["message"], "output_dir": out_dir})
        else:
            _tasks[task_id] = {"status": "error", "progress": -1,
                               "message": f"Failed (code {proc.returncode})"}
            broadcast({"type": "task_update", "task_id": task_id, "status": "error", "progress": -1})
    except Exception as e:
        _tasks[task_id] = {"status": "error", "progress": -1, "message": str(e)}
        broadcast({"type": "task_update", "task_id": task_id, "status": "error", "progress": -1})

@app.get("/api/tasks")
async def list_tasks():
    return {"tasks": [{"id": k, **v} for k, v in _tasks.items()]}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = _tasks.get(task_id)
    if not t: raise HTTPException(404)
    return {"id": task_id, **t}

@app.get("/api/outputs")
async def list_outputs():
    """List processed outputs ready for Postshot."""
    dirs = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("perspcut_"):
            images = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            dirs.append({
                "name": d.name,
                "path": str(d),
                "image_count": len(images),
                "size_mb": sum(f.stat().st_size for f in images) / 1e6 if images else 0,
            })
    return {"dirs": dirs}

# ======= WebSocket =======
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except: pass
    except WebSocketDisconnect: pass
    finally:
        if ws in _ws_clients: _ws_clients.remove(ws)

# ======= Frontend =======
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>360Cam → Postshot Dashboard</title>
<style>
:root {
  --bg: #0a0a0b; --bg2: #111113; --bg3: #18181b;
  --border: #27272a; --text: #fafafa; --text2: #a1a1aa;
  --accent: #ec4899; --accent2: #f472b6;
  --green: #22c55e; --yellow: #eab308; --red: #ef4444;
  --font: 'Inter', -apple-system, sans-serif; --mono: 'JetBrains Mono', monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.5; }
.header { padding: 20px 32px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 20px; font-weight: 600; }
.header h1 span { color: var(--accent); }
.header .subtitle { font-size: 12px; color: var(--text2); }
.container { display: flex; height: calc(100vh - 73px); }
.sidebar { width: 320px; border-right: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; }
.sidebar-section { padding: 16px; }
.sidebar-section h3 { font-size: 11px; font-weight: 600; color: var(--text2); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 12px; }
.content-item { padding: 10px 12px; border-radius: 6px; cursor: pointer; margin-bottom: 4px; transition: background .15s; }
.content-item:hover { background: var(--bg3); }
.content-item.selected { background: rgba(236,72,153,.12); border: 1px solid rgba(236,72,153,.3); }
.content-item .name { font-size: 13px; font-weight: 500; }
.content-item .meta { font-size: 11px; color: var(--text2); margin-top: 2px; }
.content-item .badge { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; }
.badge-photo { background: rgba(59,130,246,.15); color: #60a5fa; }
.badge-video { background: rgba(236,72,153,.15); color: var(--accent2); }
.main { flex: 1; padding: 24px 32px; overflow-y: auto; }
.main h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.card h3 { font-size: 12px; font-weight: 600; color: var(--text2); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 12px; }
.form-group { margin-bottom: 12px; }
.form-group label { display: block; font-size: 12px; color: var(--text2); margin-bottom: 4px; }
.form-group select, .form-group input { width: 100%; padding: 8px 10px; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 13px; }
.btn { padding: 8px 16px; border: none; border-radius: 6px; font-size: 13px; font-weight: 500; cursor: pointer; transition: all .15s; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent2); }
.btn-primary:disabled { opacity: .5; cursor: not-allowed; }
.btn-secondary { background: var(--bg3); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); }
.status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; }
.status-queued { background: rgba(234,179,8,.12); color: var(--yellow); }
.status-running { background: rgba(59,130,246,.12); color: #60a5fa; }
.status-done { background: rgba(34,197,94,.12); color: var(--green); }
.status-error { background: rgba(239,68,68,.12); color: var(--red); }
.progress-bar { height: 4px; background: var(--bg3); border-radius: 2px; margin-top: 8px; overflow: hidden; }
.progress-bar-fill { height: 100%; background: var(--accent); transition: width .3s; border-radius: 2px; }
.log-viewer { background: #000; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-family: var(--mono); font-size: 12px; color: #a1a1aa; height: 200px; overflow-y: auto; white-space: pre-wrap; }
.output-list { margin-top: 16px; }
.output-item { padding: 8px 12px; background: var(--bg3); border-radius: 6px; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; }
.output-item .name { font-size: 13px; }
.output-item .meta { font-size: 11px; color: var(--text2); }
</style>
</head>
<body>
<div class="header">
  <div><h1>360<span>Cam</span> → <span>Postshot</span></h1><div class="subtitle">360° Perspective Cut & Pipeline Dashboard</div></div>
  <div><span id="statusDisplay" class="status-badge status-queued">⏳ Idle</span></div>
</div>
<div class="container">
  <div class="sidebar">
    <div class="sidebar-section">
      <h3>📁 360 Content</h3>
      <div id="contentList"><div style="color:var(--text2);font-size:13px">Loading...</div></div>
    </div>
    <div class="sidebar-section">
      <h3>📦 Processed Outputs</h3>
      <div id="outputList"></div>
    </div>
  </div>
  <div class="main">
    <h2>Perspective Cut Tool</h2>
    <div class="grid">
      <div class="card">
        <h3>⚙️ Settings</h3>
        <div class="form-group">
          <label>Selected Input</label>
          <div id="selectedInput" style="font-size:13px;color:var(--text2);padding:8px 0">None selected</div>
        </div>
        <div class="form-group">
          <label>Preset</label>
          <select id="presetSelect">
            <option>RealityScan (default)</option>
            <option>Fisheye-like</option>
            <option>Full 360 coverage</option>
            <option>Quality (8K)</option>
            <option>Quick preview</option>
          </select>
        </div>
        <div class="form-group">
          <label>FPS (for video)</label>
          <input type="number" id="fpsInput" value="2" min="0.5" step="0.5">
        </div>
        <button class="btn btn-primary" id="runBtn" onclick="startProcessing()">▶ Run Perspective Cut</button>
      </div>
      <div class="card">
        <h3>📊 Status</h3>
        <div id="taskStatus">No active task</div>
        <div class="progress-bar" id="progressBar"><div class="progress-bar-fill" id="progressFill" style="width:0%"></div></div>
        <div class="log-viewer" id="logViewer">Waiting for task...</div>
      </div>
    </div>
    <div id="outputsSection" class="output-list"></div>
  </div>
</div>
<script>
let ws, selectedPath = null, taskId = null;

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = e => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'task_update') {
        updateTaskStatus(msg);
      }
    } catch(_) {}
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}

function updateTaskStatus(msg) {
  const statusEl = document.getElementById('taskStatus');
  const fill = document.getElementById('progressFill');
  const log = document.getElementById('logViewer');
  const globalStatus = document.getElementById('statusDisplay');
  
  const statusMap = {queued:'⏳ Queued', running:'🔵 Running', done:'✅ Done', error:'❌ Error'};
  statusEl.textContent = statusMap[msg.status] || msg.status;
  const p = Math.max(0, Math.min(100, msg.progress || 0));
  fill.style.width = p + '%';
  if (msg.message) {
    log.textContent += '\n' + msg.message;
    log.scrollTop = log.scrollHeight;
  }
  if (msg.status === 'running') {
    globalStatus.className = 'status-badge status-running';
    globalStatus.textContent = '🔵 Processing';
  } else if (msg.status === 'done') {
    globalStatus.className = 'status-badge status-done';
    globalStatus.textContent = '✅ Done';
    loadOutputs();
  } else if (msg.status === 'error') {
    globalStatus.className = 'status-badge status-error';
    globalStatus.textContent = '❌ Error';
  }
}

async function loadContent() {
  const r = await fetch('/api/content');
  const data = await r.json();
  const el = document.getElementById('contentList');
  el.innerHTML = data.items.map(i => `
    <div class="content-item" onclick="selectItem('${i.path.replace(/\\/g,'\\\\')}','${i.name}')">
      <div class="name">${i.name}</div>
      <div class="meta"><span class="badge badge-${i.type}">${i.type}</span> ${i.size_fmt} · ${i.source}</div>
    </div>
  `).join('');
}

async function loadPresets() {
  const r = await fetch('/api/presets');
  const data = await r.json();
  const sel = document.getElementById('presetSelect');
  sel.innerHTML = data.presets.map(p => `<option>${p.name}</option>`).join('');
}

async function loadOutputs() {
  const r = await fetch('/api/outputs');
  const data = await r.json();
  const el = document.getElementById('outputList');
  el.innerHTML = '<h3 style="font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">📦 Outputs</h3>';
  el.innerHTML += data.dirs.map(d => `
    <div class="output-item">
      <div><div class="name">${d.name}</div><div class="meta">${d.image_count} images · ${d.size_mb.toFixed(1)} MB</div></div>
      <div>
        <button class="btn btn-secondary" style="font-size:11px;padding:4px 8px" onclick="openFolder('${d.path.replace(/\\/g,'/')}')">📂 Open</button>
      </div>
    </div>
  `).join('');
}

function selectItem(path, name) {
  selectedPath = path;
  document.getElementById('selectedInput').innerHTML = `<strong>${name}</strong><br><span style="font-size:11px">${path}</span>`;
  document.querySelectorAll('.content-item').forEach(e => e.classList.remove('selected'));
  event.currentTarget.classList.add('selected');
}

async function startProcessing() {
  if (!selectedPath) { alert('Select a 360 file first'); return; }
  const preset = document.getElementById('presetSelect').value;
  const fps = parseFloat(document.getElementById('fpsInput').value) || null;
  const btn = document.getElementById('runBtn');
  btn.disabled = true; btn.textContent = '⏳ Processing...';
  
  const r = await fetch('/api/perspcut/start', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({input_path: selectedPath, preset, fps})
  });
  const data = await r.json();
  taskId = data.task_id;
  document.getElementById('logViewer').textContent = `Task ${taskId} started...`;
  btn.disabled = false; btn.textContent = '▶ Run Perspective Cut';
  
  // Poll for updates
  const poll = setInterval(async () => {
    const r2 = await fetch(`/api/tasks/${taskId}`);
    if (r2.ok) {
      const t = await r2.json();
      updateTaskStatus(t);
      if (t.status === 'done' || t.status === 'error') clearInterval(poll);
    }
  }, 1000);
}

function openFolder(p) { window.open(`file:///${p}`, '_blank'); }

connectWS(); loadContent(); loadPresets(); loadOutputs();
</script>
</body>
</html>
"""

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return HTMLResponse(FRONTEND_HTML)

@app.get("/api/open-folder")
async def api_open_folder(path: str = ""):
    """Open a folder in Explorer (for Postshot import)."""
    if path and os.path.exists(path):
        import subprocess
        subprocess.Popen(["explorer", path])
        return {"status": "ok"}
    return {"status": "error", "message": "Path not found"}

# ======= 360 Photo Viewer =======

PHOTO_DIR = Path(r"D:\Proyectos_Activos\2026-07-23 Le Parc Colon\360 contenido\CET COLÓN 220726-20260724T005105Z-1-001\CET COLÓN 220726\CAPTURA 360\Camera01")

@app.get("/api/360-photos")
async def list_360_photos():
    """List all 360 equirectangular photos."""
    if not PHOTO_DIR.exists():
        return {"photos": [], "count": 0}
    photos = []
    for f in sorted(PHOTO_DIR.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            sz = f.stat().st_size
            photos.append({
                "name": f.name,
                "path": str(f),
                "size_mb": round(sz / 1e6, 1),
                "url": f"/api/360-photo/{f.name}",
            })
    return {"photos": photos, "count": len(photos)}

@app.get("/api/360-photo/{filename}")
async def serve_360_photo(filename: str):
    """Serve a 360 equirectangular photo."""
    safe = Path(filename).name
    file_path = PHOTO_DIR / safe
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "Photo not found")
    return FileResponse(str(file_path), media_type="image/jpeg")

VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>360° Photo Viewer</title>
<style>
:root{--bg:#0a0a0b;--text:#fafafa;--text2:#a1a1aa;--accent:#ec4899;--accent2:#f472b6;--border:#27272a;--font:'Inter',sans-serif}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--font);background:var(--bg);color:var(--text);overflow:hidden;height:100vh}
#viewer{width:100vw;height:100vh;display:block}
#ui{position:fixed;bottom:40px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:16px;background:rgba(10,10,11,.85);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:12px;padding:12px 20px;z-index:100}
#ui .btn{background:var(--accent);color:#fff;border:none;padding:8px 16px;border-radius:8px;cursor:pointer;font-size:18px;transition:background .15s}
#ui .btn:hover{background:var(--accent2)}
#ui .photo-info{font-size:13px;color:var(--text2);text-align:center;min-width:200px}
#ui .photo-info strong{color:var(--text);display:block;font-size:14px}
#thumbstrip{position:fixed;top:20px;left:50%;transform:translateX(-50%);display:flex;gap:6px;overflow-x:auto;max-width:90vw;padding:8px;z-index:100;background:rgba(10,10,11,.7);backdrop-filter:blur(8px);border-radius:10px;border:1px solid var(--border)}
#thumbstrip img{width:60px;height:40px;object-fit:cover;border-radius:4px;cursor:pointer;opacity:.5;transition:all .2s;border:2px solid transparent}
#thumbstrip img.active{opacity:1;border-color:var(--accent)}
#thumbstrip img:hover{opacity:.8}
.hint{position:fixed;top:80px;left:50%;transform:translateX(-50%);color:var(--text2);font-size:12px;z-index:99;pointer-events:none;opacity:.6}
</style></head><body>
<div id="viewer"></div><div id="thumbstrip"></div>
<div class="hint">🖱 Arrastrá para mirar · Scroll para zoom · ◀ ▶ flechas</div>
<div id="ui">
  <button class="btn" onclick="prevPhoto()">◀</button>
  <div class="photo-info"><strong id="photoName">—</strong><span id="photoIndex">0 / 0</span></div>
  <button class="btn" onclick="nextPhoto()">▶</button>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js"}}</script>
<script type="module">
import*as THREE from'three';
import{OrbitControls}from'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';
let photos=[],currentIdx=0,scene,camera,renderer,sphere,controls;
async function loadPhotos(){const r=await fetch('/api/360-photos');const data=await r.json();photos=data.photos;
const strip=document.getElementById('thumbstrip');strip.innerHTML=photos.map((p,i)=>'<img src="'+p.url+'" class="'+(i===0?'active':'')+'" onclick="window.goTo('+i+')" title="'+p.name+'">').join('');
if(photos.length>0)loadPhoto(0);}
function loadPhoto(idx){currentIdx=idx;const photo=photos[idx];if(!photo)return;
document.getElementById('photoName').textContent=photo.name;document.getElementById('photoIndex').textContent=(idx+1)+' / '+photos.length;
document.querySelectorAll('#thumbstrip img').forEach((img,i)=>img.classList.toggle('active',i===idx));
new THREE.TextureLoader().load(photo.url,(tex)=>{tex.colorSpace=THREE.SRGBColorSpace;sphere.material.map=tex;sphere.material.needsUpdate=true});}
async function init(){scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(75,innerWidth/innerHeight,0.1,1000);camera.position.set(0,0,0.1);
renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(Math.min(devicePixelRatio,2));
document.getElementById('viewer').appendChild(renderer.domElement);
controls=new OrbitControls(camera,renderer.domElement);controls.rotateSpeed=0.5;controls.minDistance=0.1;controls.maxDistance=2;controls.enablePan=false;
const geo=new THREE.SphereGeometry(100,64,64);const mat=new THREE.MeshBasicMaterial({side:THREE.BackSide});
sphere=new THREE.Mesh(geo,mat);scene.add(sphere);
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
await loadPhotos();animate();}
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}
window.prevPhoto=()=>{if(photos.length)loadPhoto((currentIdx-1+photos.length)%photos.length)};
window.nextPhoto=()=>{if(photos.length)loadPhoto((currentIdx+1)%photos.length)};
window.goTo=(i)=>loadPhoto(i);
document.addEventListener('keydown',(e)=>{if(e.key==='ArrowLeft')window.prevPhoto();if(e.key==='ArrowRight')window.nextPhoto()});
init();
</script></body></html>
"""

@app.get("/360-viewer", include_in_schema=False)
async def serve_360_viewer():
    return HTMLResponse(VIEWER_HTML)

if __name__ == "__main__":
    import uvicorn
    port = 8766
    print(f"360Cam Dashboard: http://localhost:{port}")
    uvicorn.run("__main__:app", host="0.0.0.0", port=port, log_level="warning")


# ======= 360 Photo Viewer =======

PHOTO_DIR = Path(r"D:\Proyectos_Activos\2026-07-23 Le Parc Colon\360 contenido\CET COLÓN 220726-20260724T005105Z-1-001\CET COLÓN 220726\CAPTURA 360\Camera01")

@app.get("/api/360-photos")
async def list_360_photos():
    """List all 360 equirectangular photos."""
    if not PHOTO_DIR.exists():
        return {"photos": [], "count": 0}
    photos = []
    for f in sorted(PHOTO_DIR.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            sz = f.stat().st_size
            photos.append({
                "name": f.name,
                "path": str(f),
                "size_mb": round(sz / 1e6, 1),
                "url": f"/api/360-photo/{f.name}",
            })
    return {"photos": photos, "count": len(photos)}

@app.get("/api/360-photo/{filename}")
async def serve_360_photo(filename: str):
    """Serve a 360 equirectangular photo."""
    safe = Path(filename).name  # prevent path traversal
    file_path = PHOTO_DIR / safe
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "Photo not found")
    return FileResponse(str(file_path), media_type="image/jpeg")

# Serve the 360 viewer page
VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>360° Photo Viewer</title>
<style>
:root { --bg: #0a0a0b; --text: #fafafa; --text2: #a1a1aa; --accent: #ec4899; --accent2: #f472b6; --border: #27272a; --font: 'Inter', sans-serif; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: var(--font); background: var(--bg); color: var(--text); overflow: hidden; height: 100vh; }
#viewer { width: 100vw; height: 100vh; display: block; }
#ui { position: fixed; bottom: 40px; left: 50%; transform: translateX(-50%); display: flex; align-items: center; gap: 16px; background: rgba(10,10,11,.85); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 12px; padding: 12px 20px; z-index: 100; }
#ui .btn { background: var(--accent); color: #fff; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 18px; transition: background .15s; }
#ui .btn:hover { background: var(--accent2); }
#ui .photo-info { font-size: 13px; color: var(--text2); text-align: center; min-width: 200px; }
#ui .photo-info strong { color: var(--text); display: block; font-size: 14px; }
#thumbstrip { position: fixed; top: 20px; left: 50%; transform: translateX(-50%); display: flex; gap: 6px; overflow-x: auto; max-width: 90vw; padding: 8px; z-index: 100; background: rgba(10,10,11,.7); backdrop-filter: blur(8px); border-radius: 10px; border: 1px solid var(--border); }
#thumbstrip img { width: 60px; height: 40px; object-fit: cover; border-radius: 4px; cursor: pointer; opacity: .5; transition: all .2s; border: 2px solid transparent; }
#thumbstrip img.active { opacity: 1; border-color: var(--accent); }
#thumbstrip img:hover { opacity: .8; }
.hint { position: fixed; top: 80px; left: 50%; transform: translateX(-50%); color: var(--text2); font-size: 12px; z-index: 99; pointer-events: none; opacity: .6; }
</style>
</head>
<body>
<div id="viewer"></div>
<div id="thumbstrip"></div>
<div class="hint">🖱 Arrastrá para mirar · Scroll para zoom</div>
<div id="ui">
  <button class="btn" onclick="prevPhoto()">◀</button>
  <div class="photo-info"><strong id="photoName">—</strong><span id="photoIndex">0 / 0</span></div>
  <button class="btn" onclick="nextPhoto()">▶</button>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';

let photos = [];
let currentIdx = 0;
let scene, camera, renderer, sphere, controls;

async function loadPhotos() {
  const r = await fetch('/api/360-photos');
  const data = await r.json();
  photos = data.photos;
  
  const strip = document.getElementById('thumbstrip');
  strip.innerHTML = photos.map((p, i) =>
    `<img src="${p.url}" class="${i===0?'active':''}" onclick="window.goTo(${i})" title="${p.name}">`
  ).join('');
  
  if (photos.length > 0) loadPhoto(0);
}

function loadPhoto(idx) {
  currentIdx = idx;
  const photo = photos[idx];
  if (!photo) return;
  
  document.getElementById('photoName').textContent = photo.name;
  document.getElementById('photoIndex').textContent = `${idx+1} / ${photos.length}`;
  
  document.querySelectorAll('#thumbstrip img').forEach((img, i) => {
    img.classList.toggle('active', i === idx);
  });
  
  const texLoader = new THREE.TextureLoader();
  texLoader.load(photo.url, (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    sphere.material.map = tex;
    sphere.material.needsUpdate = true;
  });
  
  // Smooth reset rotation
  controls.target.set(0, 0, 0);
}

async function init() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 0, 0.1);
  
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  document.getElementById('viewer').appendChild(renderer.domElement);
  
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableZoom = true;
  controls.rotateSpeed = 0.5;
  controls.zoomSpeed = 0.8;
  controls.minDistance = 0.1;
  controls.maxDistance = 2;
  controls.target.set(0, 0, 0);
  controls.enablePan = false;
  
  const geo = new THREE.SphereGeometry(100, 64, 64);
  const mat = new THREE.MeshBasicMaterial({ side: THREE.BackSide });
  sphere = new THREE.Mesh(geo, mat);
  scene.add(sphere);
  
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
  
  await loadPhotos();
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

window.prevPhoto = () => { if (photos.length) loadPhoto((currentIdx - 1 + photos.length) % photos.length); };
window.nextPhoto = () => { if (photos.length) loadPhoto((currentIdx + 1) % photos.length); };
window.goTo = (i) => loadPhoto(i);

// Keyboard support
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') window.prevPhoto();
  if (e.key === 'ArrowRight') window.nextPhoto();
});

init();
</script>
</body>
</html>
"""

@app.get("/360-viewer", include_in_schema=False)
async def serve_360_viewer():
    return HTMLResponse(VIEWER_HTML)
