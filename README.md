# Visor 360 Interactivo — Pipeline 3D Gaussian Splatting

Sistema completo para capturar, procesar y visualizar entornos 360° con **3D Gaussian Splatting**. Desde fotos/videos Insta360 hasta modelos 3D navegables, con dashboards web profesionales.

## 🔗 Acceso Rápido

| Servicio | URL | Puerto |
|---|---|---|
| **Dashboard Principal** | http://localhost:8080 | 8080 |
| **Teatro Colón GS** | http://localhost:8765 | 8765 |
| **360Cam → Postshot** | http://localhost:8766 | 8766 |
| **Visor 360 Interactivo** | http://localhost:8766/360-viewer | 8766 |

---

## 🧠 Stack Tecnológico

| Herramienta | Rol |
|---|---|
| **Python 3.13** | Entorno principal |
| **PyTorch 2.6.0+cu124** | Backend CUDA para training |
| **gsplat 1.5.3** | Gaussian Splat rasterization |
| **COLMAP 4.1.1** | Structure-from-Motion (GPU SIFT) |
| **FFmpeg 8.0** | Procesamiento de video 360° |
| **360Cam-PGM-3DGS-Tools** | Conversión equirectangular → perspectiva |
| **FastAPI** | Backend REST + WebSocket |
| **Three.js** | Visores 3D (Gaussian Splats + 360°) |
| **Jawset Postshot** | Entrenamiento de Gaussian Splats (GUI) |

---

## 📊 Dashboards

### 1. Teatro Colón GS (puerto 8765)
Dashboard para el pipeline de 8 videos del Teatro Colón convertidos a nubes de puntos coloreadas.

- Sidebar con lista de videos y estados
- Log viewer en vivo
- Visor 3D con Three.js
- 8 modelos .ply con colores RGB reales

### 2. 360Cam → Postshot (puerto 8766)
Dashboard para convertir capturas 360° en datasets listos para Postshot:

- **Visor 360°** — Navegación interactiva de fotos equirectangulares dentro de una esfera
- **Content Browser** — Exploración de archivos .insv, .jpg, .mp4
- **360PerspCut** — Conversión de 360° a imágenes perspectiva con presets configurables
- **Output Manager** — Visualización de datasets generados

---

## 🔄 Pipeline 360° → Gaussian Splat

```
📷 Insta360
   │
   ├── .jpg (equirectangular 360)
   │       │
   │       ▼
   │   360PerspCut (8 vistas, 90° FOV, 1600px)
   │       │
   │       ▼
   │   📁 664 imágenes perspectiva (238 MB)
   │
   ├── .insv (dual-fisheye)
   │       │
   │       ▼
   │   ffmpeg v360=dfisheye→equirect
   │       │
   │       ▼
   │   📁 equirect_360.mp4 (4096×2048, H.265)
   │       │
   │       ▼
   │   360PerspCut (1 fps, 8 vistas)
   │       │
   │       ▼
   │   📁 2,872 imágenes perspectiva (764 MB)
   │
   └── ▶ Jawset Postshot
           │
           ▼
       🎯 3D Gaussian Splatting (.ply / .spz)
```

### Configuración Recomendada para Postshot

| Parámetro | Valor |
|---|---|
| Camera Poses | Compute From Images |
| Pose Quality | 4 (Best) |
| Single Lens & Focal Length | ✅ **Marcado** |
| Radiance Field Profile | Splat3 |
| Max Splat Count | 300000 |
| Stop Training | 100 kSteps |
| Anti-Aliasing | ✅ |

---

## 📁 Estructura del Proyecto

```
Teatro_Colon-GS/
├── backend/                    # Pipeline Python
│   ├── pipeline.py             # Orquestador principal
│   ├── extract_frames.py       # Extracción de frames
│   ├── run_colmap.py           # COLMAP SfM
│   ├── train_gs.py             # Training + export PLY
│   └── server.py               # FastAPI backend (puerto 8765)
├── frontend/
│   └── index.html              # Dashboard Teatro Colón (1882 líneas)
├── Postshot/                   # Datasets para Postshot
│   ├── test_perspcut/          # 664 imágenes (fotos 360)
│   ├── perspcut_video_360/     # 2,872 imágenes (video)
│   └── equirect_360.mp4        # Video equirectangular (0.9 GB)
├── gs_output/                  # Modelos .ply generados
├── colmap_input/               # Reconstrucciones COLMAP
├── colmap-bin/                 # COLMAP 4.1.1 binario (CUDA)
├── venv/                       # Entorno Python
├── config/                     # Configuración YAML
└── 360Cam-PGM-3DGS-Tools/     # Herramientas 360°
    └── webapp/server.py        # Backend 360° (puerto 8766)
```

---

## 🚀 Inicio Rápido

### Prerrequisitos
- Windows 10+ con RTX 4090 (24GB VRAM recomendado)
- Python 3.13
- FFmpeg en PATH
- Jawset Postshot (gratuito)

### Instalación

```bash
# Clonar
git clone https://github.com/tolchx/visor-360-interactivo.git
cd visor-360-interactivo

# Entorno Python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Iniciar servidores
python backend\server.py          # Dashboard Teatro Colón (8765)
python webapp\server.py           # Dashboard 360° (8766)
```

### Conversión Rápida (fotos 360 → Postshot)

```bash
# 1. Convertir fotos equirectangulares a perspectiva
python cli_tools/gs360_360PerspCut.py \
  -i "ruta/a/tus/fotos/" \
  -o "output/" \
  --count 8 --size 1600 --hfov 90 --focal-mm 12

# 2. Arrastrar carpeta "output/" a Postshot
# 3. Configurar: Single Lens & Focal Length = ON, 300k splats, 100k steps
```

### Conversión de Video .insv (Insta360)

```bash
# 1. Convertir dual-fisheye a equirectangular
ffmpeg -i video.insv \
  -vf "v360=dfisheye:output=e:ih_fov=190:iv_fov=190:w=4096:h=2048" \
  -c:v libx265 -preset fast -crf 28 -an equirect_360.mp4

# 2. Extraer frames perspectiva
python cli_tools/gs360_360PerspCut.py \
  -i equirect_360.mp4 -o output_video/ \
  --count 8 --size 1600 --hfov 90 --focal-mm 12 -f 1
```

---

## 🎯 Aplicaciones

- **Preservación arquitectónica** — Escaneo 3D de teatros, museos, espacios patrimoniales
- **Tour virtual inmersivo** — Experiencia 360° navegable para clientes
- **Gemelo digital** — Modelo 3D realista para VR/AR
- **Integración Unreal Engine** — Exportación directa desde Postshot

---

## 📄 Licencia

MIT — Ver archivo [LICENSE](LICENSE)

## 👤 Autor

**Tolch (Luciano Toledo)** — [tolchx.com](https://tolchx.com)
