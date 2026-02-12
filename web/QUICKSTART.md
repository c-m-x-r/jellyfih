# Jellyfih Web Viewer - Quick Start

> **Note:** The web viewer works standalone with pip. You don't need uv or the full simulation suite!

## One-Liner Setup

```bash
cd web/ && ./install.sh && python app.py
```

Then open: **http://localhost:5000**

---

## Step-by-Step

### 1. Install Dependencies (Simple Method)
```bash
cd web/
./install.sh
```

### Alternative: Manual Installation
```bash
pip install flask numpy matplotlib scipy pillow
```

### 2. Check Installation
```bash
cd web/
python check_deps.py
```

### 3. Test API (Optional)
```bash
python test_api.py
```

### 4. Start Server
```bash
python app.py
```

### 5. Open Browser
- Local: http://localhost:5000
- Termux: http://127.0.0.1:5000

---

## Usage

### Preset Genomes
- **Default** - Midpoint of all parameter bounds
- **Aurelia** - Hand-designed moon jelly reference
- **Random** - Random genome within bounds

### Editing
1. Drag sliders to adjust 9 genome parameters
2. Watch morphology update in real-time (300ms debounce)
3. See statistics update automatically

### Copy/Paste
1. Click **Copy** to copy genome JSON to clipboard
2. Paste into textarea and click **Load** to restore

---

## Troubleshooting

### "Module not found" error
```bash
# Install missing module
pip install <module_name>
```

### Port already in use
```bash
# Kill existing process
pkill -f "python app.py"

# Or use different port
# Edit app.py line: app.run(port=5001)
```

### Cannot access from other devices
```bash
# Make sure Flask is listening on 0.0.0.0
# Already configured in app.py:
# app.run(host='0.0.0.0', port=5000)
```

### Rendering is slow
- Normal: 200-500ms for 80K particles
- If >2s: Check CPU/memory usage
- Reduce grid_res in app.py (line 14) to 64 for faster preview

---

## Files Overview

```
web/
├── app.py           # Flask server (START HERE)
├── static/
│   ├── style.css    # Brutalist design
│   └── app.js       # Slider logic, API calls
├── templates/
│   └── index.html   # Main page
├── README.md        # Full documentation
├── DESIGN.md        # Design specification
└── run.sh           # Startup script
```

---

## Common Tasks

**Change render resolution:**
Edit `app.py` line 14:
```python
GRID_RES = 64  # Faster, lower quality
GRID_RES = 128 # Default
GRID_RES = 256 # Slower, higher quality
```

**Change particle count:**
Edit `app.py` line 13:
```python
N_PARTICLES = 40000  # Faster
N_PARTICLES = 80000  # Default
```

**Change color scheme:**
Edit `web/static/style.css` lines 18-24 (`:root` variables)

---

## Keyboard Shortcuts (Future)

- `Tab` - Navigate controls
- `Arrow keys` - Adjust focused slider
- `R` - Random genome
- `A` - Load Aurelia
- `D` - Load Default
- `Ctrl+Z` - Undo (not implemented)
- `Ctrl+C` - Copy genome (not implemented)

---

**Ready to explore!** 🚀
