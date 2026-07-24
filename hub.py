"""Main Dashboard — Hub for all 360/GS services."""
import subprocess, sys, os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Visor 360 — Hub", version="2.0.0")

HUB_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visor 360 Interactivo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #060608; --bg2: #0c0c10; --bg3: #141418; --bg4: #1c1c22;
  --border: #22222a; --border-hover: #2a2a34;
  --text: #f0f0f2; --text2: #90909a; --text3: #60606a;
  --accent: #ec4899; --accent2: #f472b6; --accent-dim: rgba(236,72,153,.12);
  --blue: #3b82f6; --blue-dim: rgba(59,130,246,.12);
  --green: #22c55e; --green-dim: rgba(34,197,94,.12);
  --amber: #f59e0b; --amber-dim: rgba(245,158,11,.12);
  --radius: 12px; --radius-sm: 8px;
  --font: 'Inter', system-ui, -apple-system, sans-serif;
  --mono: 'JetBrains Mono', 'SF Mono', monospace;
  --transition: 250ms cubic-bezier(.16,1,.3,1);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; min-height: 100vh; }

/* Header */
.header { padding: 40px 48px 32px; max-width: 1280px; margin: 0 auto; }
.header h1 { font-size: 36px; font-weight: 800; letter-spacing: -.03em; }
.header h1 span { color: var(--accent); }
.header .sub { color: var(--text2); font-size: 15px; margin-top: 6px; font-weight: 400; }
.header .tagline { color: var(--text3); font-size: 13px; margin-top: 4px; }

/* Grid */
.container { max-width: 1280px; margin: 0 auto; padding: 0 48px 48px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }

/* Cards */
.card {
  background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 0; overflow: hidden; transition: all var(--transition);
}
.card:hover { border-color: var(--border-hover); transform: translateY(-2px); }
.card-header {
  padding: 24px 24px 16px; display: flex; align-items: flex-start; gap: 16px;
}
.card-icon {
  width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center;
  justify-content: center; font-size: 20px; flex-shrink: 0;
}
.card-icon.purple { background: var(--accent-dim); }
.card-icon.blue { background: var(--blue-dim); }
.card-icon.green { background: var(--green-dim); }
.card-icon.amber { background: var(--amber-dim); }
.card-info h3 { font-size: 16px; font-weight: 600; }
.card-info .url { font-size: 13px; color: var(--text3); font-family: var(--mono); margin-top: 2px; }
.card-info .desc { font-size: 13px; color: var(--text2); margin-top: 6px; line-height: 1.5; }
.card-status {
  display: flex; align-items: center; gap: 6px; margin-top: 10px;
  font-size: 12px; font-weight: 500;
}
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot.green { background: var(--green); }
.dot.amber { background: var(--amber); }
.card-footer {
  padding: 12px 24px; border-top: 1px solid var(--border);
  display: flex; gap: 8px; flex-wrap: wrap;
}
.btn {
  padding: 7px 16px; border-radius: var(--radius-sm); font-size: 13px; font-weight: 500;
  cursor: pointer; border: none; transition: all var(--transition); text-decoration: none;
  display: inline-flex; align-items: center; gap: 6px;
}
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent2); }
.btn-secondary { background: var(--bg4); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); }
.btn-open { background: transparent; color: var(--text2); border: 1px solid var(--border); }
.btn-open:hover { background: var(--bg4); color: var(--text); }

/* Stats row */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 24px; }
.stat-card {
  background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: 16px; text-align: center;
}
.stat-card .num { font-size: 28px; font-weight: 700; color: var(--accent); }
.stat-card .label { font-size: 12px; color: var(--text3); margin-top: 2px; text-transform: uppercase; letter-spacing: .05em; }

/* Section title */
.section-title { font-size: 13px; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: .08em; margin: 32px 0 16px; }

/* Log */
.log-bar {
  margin-top: 32px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: 14px 18px; font-family: var(--mono); font-size: 12px; color: var(--text3);
  max-height: 200px; overflow-y: auto; white-space: pre-wrap;
}
.log-bar .time { color: var(--text2); }

/* Responsive */
@media (max-width: 720px) {
  .header { padding: 24px 20px 20px; }
  .container { padding: 0 20px 32px; }
  .grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<div class="header">
  <h1>360<span>Visor</span></h1>
  <div class="sub">Pipeline 3D Gaussian Splatting · Captura 360° → Modelo 3D</div>
  <div class="tagline">Insta360 · FFmpeg · COLMAP · gsplat · Postshot · Three.js</div>
</div>

<div class="container">

  <!-- Stats -->
  <div class="stats">
    <div class="stat-card"><div class="num">83</div><div class="label">Fotos 360</div></div>
    <div class="stat-card"><div class="num">2,872</div><div class="label">Imágenes Perspectiva</div></div>
    <div class="stat-card"><div class="num">8</div><div class="label">Nubes de Puntos</div></div>
    <div class="stat-card"><div class="num">1</div><div class="label">Training Activo</div></div>
  </div>

  <!-- Apps -->
  <div class="section-title">📱 Aplicaciones</div>
  <div class="grid">

    <div class="card">
      <div class="card-header">
        <div class="card-icon purple">🌐</div>
        <div class="card-info">
          <h3>Visor 360° Interactivo</h3>
          <div class="url">/360-viewer · :8766</div>
          <div class="desc">Navegación inmersiva de 83 fotos equirectangulares dentro de una esfera 3D. Arrastrá para mirar, scroll para zoom, flechas para cambiar de foto.</div>
          <div class="card-status"><span class="dot green"></span> 83 fotos cargadas</div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" href="http://localhost:8766/360-viewer" target="_blank">Abrir Visor 360°</a>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-icon purple">🎯</div>
        <div class="card-info">
          <h3>360Cam → Postshot</h3>
          <div class="url">:8766</div>
          <div class="desc">Conversión de capturas 360° a datasets perspectiva, con presets configurables. Monitoreo de tareas y gestión de outputs para Postshot.</div>
          <div class="card-status"><span class="dot green"></span> 2 datasets listos</div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" href="http://localhost:8766" target="_blank">Abrir Dashboard</a>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-icon blue">🏛️</div>
        <div class="card-info">
          <h3>Teatro Colón GS</h3>
          <div class="url">:8765</div>
          <div class="desc">8 salones del Teatro Colón convertidos a nubes de puntos coloreadas con pipeline COLMAP + gsplat. Visor 3D integrado.</div>
          <div class="card-status"><span class="dot green"></span> 8 modelos listos</div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn btn-primary" href="http://localhost:8765" target="_blank">Abrir Dashboard</a>
      </div>
    </div>

  </div>

  <!-- Datasets -->
  <div class="section-title">📦 Datasets Listos para Postshot</div>
  <div class="grid">

    <div class="card">
      <div class="card-header">
        <div class="card-icon amber">📷</div>
        <div class="card-info">
          <h3>Fotos 360 — Le Parc Colon</h3>
          <div class="desc">83 fotos Insta360 → 664 imágenes perspectiva (8 vistas · 1600px · 90° FOV)</div>
          <div class="card-status"><span class="dot green"></span> 238 MB · Training en progreso</div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn-secondary" href="#" onclick="alert('Postshot training en progreso (~10hs)')">Ver Progreso</a>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-icon amber">🎬</div>
        <div class="card-info">
          <h3>Video 360 — Recorrido</h3>
          <div class="desc">Video .insv 6min → 2,872 imágenes perspectiva (359 frames × 8 vistas)</div>
          <div class="card-status"><span class="dot amber"></span> 764 MB · Pendiente</div>
        </div>
      </div>
      <div class="card-footer">
        <a class="btn-secondary" href="file:///C:/Users/Tolch/Documents/AI_Code/Teatro_Colon-GS/Postshot/perspcut_video_360" target="_blank">Abrir Carpeta</a>
      </div>
    </div>

  </div>

  <!-- Pipeline Status -->
  <div class="section-title">⚙️ Estado del Sistema</div>
  <div id="statusLog" class="log-bar">Conectando...</div>

</div>

<script>
async function loadStatus() {
  const log = document.getElementById('statusLog');
  const services = [
    {name:'Teatro Colón GS', port:8765},
    {name:'360Cam Dashboard', port:8766},
    {name:'Visor 360', port:8766, path:'/360-viewer'},
  ];
  
  let lines = [];
  const now = new Date();
  lines.push(`[${now.toLocaleTimeString()}] Verificando servicios...`);
  
  for (const svc of services) {
    try {
      const r = await fetch(`http://localhost:${svc.port}/api/health`, {signal: AbortSignal.timeout(3000)});
      if (r.ok) {
        lines.push(`  ✅ ${svc.name} — http://localhost:${svc.port} — Activo`);
      } else {
        lines.push(`  ⚠️  ${svc.name} — http://localhost:${svc.port} — Respuesta inesperada`);
      }
    } catch {
      lines.push(`  ❌ ${svc.name} — http://localhost:${svc.port} — No responde`);
    }
  }
  
  // Check 360 API
  try {
    const r = await fetch('http://localhost:8766/api/360-photos', {signal: AbortSignal.timeout(3000)});
    if (r.ok) {
      const d = await r.json();
      lines.push(`  📷 ${d.count} fotos 360 disponibles en el visor`);
    }
  } catch {}
  
  // Check outputs
  try {
    const r = await fetch('http://localhost:8766/api/outputs', {signal: AbortSignal.timeout(3000)});
    if (r.ok) {
      const d = await r.json();
      for (const o of d.dirs) {
        lines.push(`  📦 ${o.name}: ${o.image_count} imágenes, ${o.size_mb.toFixed(0)} MB`);
      }
    }
  } catch {}
  
  log.textContent = lines.join('\n');
}

loadStatus();
setInterval(loadStatus, 15000);
</script>
</body>
</html>
"""

@app.get("/", include_in_schema=False)
async def serve_hub():
    return HTMLResponse(HUB_HTML)

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Visor 360 Hub", "version": "2.0.0"}

if __name__ == "__main__":
    port = 8080
    print(f"🌐 Visor 360 Hub: http://localhost:{port}")
    uvicorn.run("__main__:app", host="0.0.0.0", port=port, log_level="warning")
