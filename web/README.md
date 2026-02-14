# Jellyfih Web Viewer

Minimalist, responsive web interface for exploring jellyfish morphologies.

## Design Philosophy

- **Brutalist/technical aesthetic** - Clean, functional, no unnecessary ornamentation
- **Minimal color palette** - Grayscale base with subtle accent colors
- **Typography-focused** - System fonts, monospace for data
- **Mobile-first** - Responsive grid layout that works on Termux
- **Fast and lightweight** - No JavaScript frameworks, vanilla implementations

## Features

- **Real-time genome editing** - 9 sliders for morphology parameters
- **Live morphology preview** - Server-rendered particle visualization
- **Statistics panel** - Particle counts and morphology metrics
- **Preset genomes** - Default, Aurelia (moon jelly), Random
- **Import/Export** - Copy/paste genome JSON

## Installation

### Dependencies

```bash
# Core dependencies
pip install flask numpy matplotlib scipy

# Or if using uv (recommended)
uv pip install flask numpy matplotlib scipy
```

### Quick Start

```bash
# Navigate to web directory
cd web/

# Run Flask app
python app.py

# Access at http://localhost:5000
# On Termux: http://127.0.0.1:5000
```

### Production Deployment

For production use, consider using gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### GET /
Main viewer interface

### POST /api/render
Render morphology from genome

**Request:**
```json
{
  "genome": [0.125, 0.0, 0.15, -0.025, 0.2, -0.24, 0.05, 0.0625, 0.025]
}
```

**Response:**
```json
{
  "image": "data:image/png;base64,...",
  "stats": {
    "n_total": 80000,
    "n_robot": 15234,
    "n_jelly": 12500,
    "muscle_count": 2734,
    "n_water": 64516
  }
}
```

### GET /api/random
Generate random genome

### GET /api/aurelia
Get Aurelia aurita reference genome

### GET /api/default
Get default genome (midpoint of bounds)

### GET /api/bounds
Get genome parameter bounds

## Architecture

```
web/
├── app.py              # Flask backend
├── static/
│   ├── style.css       # Minimalist CSS (brutalist aesthetic)
│   └── app.js          # Vanilla JavaScript (no frameworks)
└── templates/
    └── index.html      # Single-page interface
```

## Design Specifications

### Color Palette

```
Background:  #FAFAFA (off-white)
Foreground:  #1A1A1A (near-black)
Border:      #E0E0E0 (light gray)
Accent:      #4ECDC4 (teal)
Muted:       #757575 (gray)
Highlight:   #FF6B6B (coral)
```

### Material Colors

```
Water:    #E8F4F8 (light blue-gray)
Jelly:    #4ECDC4 (teal)
Payload:  #FFA500 (orange)
Muscle:   #FF6B6B (coral)
```

### Typography

- **UI:** System font stack (-apple-system, BlinkMacSystemFont, Segoe UI, Roboto)
- **Data:** Monospace (SF Mono, Monaco, Inconsolata, Courier New)
- **Scale:** 16px base, modular scale for headings

### Responsive Breakpoints

- **Mobile (<768px):** Single column, stacked panels
- **Tablet (768-1024px):** Two column (controls left, viewer/stats right)
- **Desktop (>1200px):** Three column (controls | viewer | stats)

## Performance

- **Debounced rendering:** 300ms delay after slider adjustment
- **Server-side rendering:** Uses existing `make_jelly.py` for accuracy
- **Typical render time:** 200-500ms for 80K particles at 128x128 grid

## Browser Compatibility

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile browsers (tested on Termux)
- Requires JavaScript enabled
- Canvas API support required

## Development

### File Structure

- **app.py:** Flask routes and rendering logic
- **style.css:** Complete styling (no external CSS frameworks)
- **app.js:** UI logic, API calls, state management
- **index.html:** HTML structure

### Adding Features

1. **New API endpoint:** Add route in `app.py`
2. **UI component:** Update `index.html` and `style.css`
3. **Interaction:** Add logic in `app.js`

### Debugging

```bash
# Run with debug mode
FLASK_DEBUG=1 python app.py

# Check logs
tail -f /tmp/flask.log
```

## Future Enhancements

- [ ] Client-side Bezier preview (instant feedback)
- [ ] WebSocket for real-time updates
- [ ] Gallery page with saved morphologies
- [ ] Comparison mode (side-by-side)
- [ ] Undo/redo history
- [ ] Keyboard shortcuts
- [ ] Export as SVG/PDF
- [ ] Dark mode toggle

## License

Part of the Jellyfih project - evolutionary soft robotics research.
