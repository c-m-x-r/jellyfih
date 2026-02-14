# Frontend Genome Builder/Viewer Specification

## Overview
A web-based interactive tool for exploring jellyfish morphologies, understanding the genotype-phenotype mapping, and experimenting with genome parameters in real-time.

## Target Audience
- **Primary**: Computational biologists and researchers
- **Secondary**: Students learning about soft robotics
- **Tertiary**: General public interested in evolutionary algorithms

## Technology Stack
- **Backend**: Flask (lightweight Python web framework)
- **Visualization**: Matplotlib for static renders, potentially Three.js for interactive 3D
- **Frontend**: HTML5 + JavaScript (vanilla or Vue.js for reactivity)
- **Communication**: REST API + WebSocket for real-time updates

## Package Structure
```
jellyfih/
├── jellys/
│   └── web/                     # New web interface module
│       ├── __init__.py
│       ├── app.py              # Flask application
│       ├── api.py              # REST API endpoints
│       ├── static/
│       │   ├── css/
│       │   │   └── style.css
│       │   ├── js/
│       │   │   ├── genome-editor.js
│       │   │   ├── morphology-viewer.js
│       │   │   └── examples.js
│       │   └── img/
│       ├── templates/
│       │   ├── index.html      # Main genome builder page
│       │   ├── gallery.html    # Example morphologies
│       │   └── education.html  # Genotype-phenotype explanation
│       └── renderer.py         # Server-side morphology rendering
```

---

## Core Features

### 1. Interactive Genome Editor

**UI Components:**
- 9 sliders (one per gene) with real-time value display
- Each slider has:
  - Label with gene name and description
  - Min/max bounds from `GENOME_LOWER` and `GENOME_UPPER`
  - Current value display with 3 decimal precision
  - Reset to default button
- "Randomize" button to generate random genome
- "Load Aurelia" button to load moon jelly reference
- "Copy Genome" button to copy JSON to clipboard
- "Paste Genome" text area to load saved genomes

**Gene Descriptions** (for UI labels):
```javascript
const GENE_INFO = {
  0: {
    name: "CP1 X-offset",
    description: "Control point 1 horizontal position (bell curvature)",
    bounds: [0.0, 0.25],
    unit: "normalized",
    biologicalContext: "Affects bell width near payload attachment"
  },
  1: {
    name: "CP1 Y-offset",
    description: "Control point 1 vertical position",
    bounds: [-0.15, 0.15],
    unit: "normalized",
    biologicalContext: "Controls bell profile convexity/concavity"
  },
  2: {
    name: "CP2 X-offset",
    description: "Control point 2 horizontal position",
    bounds: [0.0, 0.3],
    unit: "normalized",
    biologicalContext: "Determines bell flare angle"
  },
  3: {
    name: "CP2 Y-offset",
    description: "Control point 2 vertical position",
    bounds: [-0.2, 0.15],
    unit: "normalized",
    biologicalContext: "Shapes mid-bell contour"
  },
  4: {
    name: "Bell Tip X",
    description: "Bell tip horizontal extent (bell radius)",
    bounds: [0.05, 0.35],
    unit: "normalized",
    biologicalContext: "Overall bell diameter"
  },
  5: {
    name: "Bell Tip Y",
    description: "Bell tip vertical extent (bell height)",
    bounds: [-0.45, -0.03],
    unit: "normalized",
    biologicalContext: "Bell cavity depth"
  },
  6: {
    name: "Thickness (Base)",
    description: "Bell thickness at payload attachment",
    bounds: [0.025, 0.08],
    unit: "normalized",
    biologicalContext: "Structural strength at root"
  },
  7: {
    name: "Thickness (Mid)",
    description: "Bell thickness at mid-section",
    bounds: [0.025, 0.1],
    unit: "normalized",
    biologicalContext: "Swimming muscle mass"
  },
  8: {
    name: "Thickness (Tip)",
    description: "Bell thickness at outer edge",
    bounds: [0.01, 0.04],
    unit: "normalized",
    biologicalContext: "Bell margin flexibility"
  }
};
```

### 2. Real-Time Morphology Viewer

**Visualization Canvas:**
- 600x800px canvas showing jellyfish morphology
- Color-coded materials:
  - Blue: Water (background)
  - Cyan/Turquoise: Jelly (mesoglea)
  - Red: Payload (instrumentation)
  - Yellow/Orange: Muscle tissue
- Grid overlay with scale markers
- Spawn position indicator ([0.5, 0.7])
- Bounding box showing tank dimensions

**Update Strategy:**
- **Option A (Server-side)**: Send genome → Flask renders PNG → display in img tag
  - Pros: Uses existing `make_jelly.py` code, accurate
  - Cons: Network latency (~200-500ms)
- **Option B (Client-side)**: Implement Bezier curve generation in JavaScript
  - Pros: Instant feedback (<16ms)
  - Cons: Need to port Python logic to JS, potential inconsistencies
- **Recommended**: Option A initially, optimize to Option B later

**API Endpoint:**
```python
# jellys/web/api.py
@app.route('/api/render', methods=['POST'])
def render_genome():
    """Render morphology from genome.

    Request body:
    {
        "genome": [0.125, 0.0, 0.15, -0.025, 0.2, -0.24, 0.05, 0.0625, 0.025],
        "render_mode": "particles" | "outline",
        "colormap": "materials" | "velocity"
    }

    Returns:
    {
        "image": "data:image/png;base64,...",
        "stats": {
            "n_robot": 15234,
            "n_jelly": 12500,
            "muscle_count": 2734,
            "n_water": 60000
        }
    }
    """
```

### 3. Phenotype Statistics Panel

Display computed stats from `fill_tank()`:
```
╔══════════════════════════════════════╗
║ PHENOTYPE STATISTICS                 ║
╠══════════════════════════════════════╣
║ Total Particles:        80,000       ║
║ Robot Particles:        15,234 (19%) ║
║   - Jelly (Mesoglea):   12,500       ║
║   - Muscle Tissue:       2,734       ║
║   - Payload:               250       ║
║ Water Particles:        64,516 (81%) ║
║ Dead (Padding):            250       ║
╠══════════════════════════════════════╣
║ Morphology Metrics                   ║
║ Bell Diameter:           0.40 units  ║
║ Bell Height:             0.42 units  ║
║ Muscle/Jelly Ratio:      0.22        ║
║ Aspect Ratio (H/D):      1.05        ║
╚══════════════════════════════════════╝
```

### 4. Example Morphologies Gallery

**Gallery Page** showing:
1. **Aurelia aurita** (moon jelly reference)
   - Genome display
   - Fitness score
   - Description: "Hand-designed biomimetic baseline"
2. **Best Evolved Morphology** (from `best_genomes.json`)
   - Genome from highest fitness generation
   - Fitness score and generation number
   - Description: "Optimized for staying power"
3. **Generation Progression** (4 thumbnails)
   - Gen 0, Gen 16, Gen 33, Gen 49
   - Show evolutionary trajectory
4. **Interesting Extremes**
   - Thickest bell
   - Tallest bell
   - Widest bell
   - Minimal muscle

**Gallery Item Card:**
```html
<div class="morphology-card">
  <img src="/api/render?genome=[...]" alt="Morphology">
  <h3>Aurelia aurita</h3>
  <p>Fitness: 0.6542</p>
  <p>Generation: Reference</p>
  <button onclick="loadGenome([...])">Load in Editor</button>
  <button onclick="copyGenome([...])">Copy Genome</button>
</div>
```

### 5. Educational Content: Genotype-Phenotype Mapping

**Interactive Tutorial Page** with sections:

**Section 1: What is a Genome?**
```
A genome is a 9-dimensional vector that encodes the shape of a jellyfish bell.
Each gene controls a specific aspect of the morphology through Bezier curve
control points and thickness parameters.

[Diagram showing genes → Bezier curve → 3D bell shape]
```

**Section 2: Bezier Curve Basics**
```
The bell profile is generated using a cubic Bezier curve:
- Start point: Payload attachment (0, 0)
- Control point 1: Genes 0-1 (cp1_x, cp1_y)
- Control point 2: Genes 2-3 (cp2_x, cp2_y)
- End point: Genes 4-5 (end_x, end_y)

[Interactive animation: drag control points, see curve update]
```

**Section 3: From Curve to Bell**
```
The 2D Bezier curve is:
1. Sampled at regular intervals
2. Thickened using genes 6-8 (t_base, t_mid, t_tip)
3. Mirrored horizontally for symmetry
4. Filled with particles (jelly, muscle, payload)

[Step-by-step visualization]
```

**Section 4: Material Layers**
```
Each bell has structural layers:
- Inner layer: Muscle tissue (active contraction)
- Middle layer: Mesoglea (passive jelly)
- Outer collar: Muscle ring for structural integrity

[Cross-section diagram with material annotations]
```

**Section 5: Try It Yourself**
```
Experiment with these genomes to see their effects:
- Wide bell: Increase gene 4 (end_x) → 0.35
- Tall bell: Decrease gene 5 (end_y) → -0.45
- Thick muscle: Increase gene 7 (t_mid) → 0.1

[Link to genome editor with preset buttons]
```

### 6. Advanced Features

**Compare Mode:**
- Split-screen showing two morphologies side-by-side
- Diff highlighting changed genes
- Use case: Compare evolved vs. Aurelia, or before/after edits

**Simulation Preview** (stretch goal):
- Run short (5-second) simulation with actuation
- Show altitude over time plot
- Estimate fitness without full 60K-step evaluation
- Requires WASM build of Taichi or server-side GPU queue

**Export Options:**
- Export genome as JSON file
- Export morphology as PNG/SVG
- Export to simulation config for batch runs

---

## API Specification

**REST Endpoints:**

```python
# GET /api/examples
# Returns: List of example genomes (Aurelia, best evolved, etc.)
{
  "examples": [
    {
      "name": "Aurelia aurita",
      "genome": [0.125, 0.0, 0.15, ...],
      "fitness": 0.6542,
      "generation": "reference",
      "description": "Hand-designed moon jelly"
    },
    ...
  ]
}

# POST /api/render
# Body: { "genome": [...], "render_mode": "particles", "colormap": "materials" }
# Returns: { "image": "data:image/png;base64,...", "stats": {...} }

# POST /api/validate
# Body: { "genome": [...] }
# Returns: { "valid": true, "warnings": [], "errors": [] }
# Validates genome against bounds, checks for degenerate morphologies

# GET /api/best_genomes
# Returns: Contents of output/best_genomes.json (evolution history)

# POST /api/compare
# Body: { "genome_a": [...], "genome_b": [...] }
# Returns: { "image_a": "...", "image_b": "...", "diff": {...} }
```

**WebSocket (optional, for real-time updates):**
```javascript
ws = new WebSocket('ws://localhost:5000/ws/render');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateViewer(data.image, data.stats);
};
// Send genome updates as user drags sliders
```

---

## UI Mockup (Text-based)

```
┌─────────────────────────────────────────────────────────────────┐
│ JELLYFIH: Interactive Genome Builder                      [?][X]│
├────────────────────┬────────────────────────────────────────────┤
│ GENOME EDITOR      │ MORPHOLOGY VIEWER                          │
│                    │                                            │
│ CP1 X-offset       │    ╔════════════════════════════════╗     │
│ [──●──────] 0.125  │    ║          ^                     ║     │
│ Control point 1    │    ║          │                     ║     │
│ horizontal         │    ║     ┌────┴────┐                ║     │
│                    │    ║     │ Payload │                ║     │
│ CP1 Y-offset       │    ║     └────┬────┘                ║     │
│ [────●────] 0.000  │    ║      ╱   │   ╲                 ║     │
│                    │    ║    ╱   ╱─┴─╲   ╲               ║     │
│ CP2 X-offset       │    ║   │  ╱Muscle╲  │              ║     │
│ [───●─────] 0.150  │    ║   │ │  Jelly  │ │              ║     │
│                    │    ║   │ │         │ │              ║     │
│ CP2 Y-offset       │    ║    ╲ ╲_______╱ ╱               ║     │
│ [─────●───] -0.025 │    ║      ╲       ╱                 ║     │
│                    │    ║        ╲   ╱                   ║     │
│ [... 5 more genes] │    ║          V                     ║     │
│                    │    ╚════════════════════════════════╝     │
│ [Randomize]        │                                            │
│ [Load Aurelia]     │ ╔══════════════════════════════════════╗  │
│ [Reset All]        │ ║ PHENOTYPE STATS                      ║  │
│                    │ ╠══════════════════════════════════════╣  │
│ Paste Genome:      │ ║ Total Particles:      80,000         ║  │
│ ┌────────────────┐ │ ║ Robot Particles:      15,234 (19%)   ║  │
│ │[0.125,0.0,...] │ │ ║ Muscle Count:          2,734         ║  │
│ └────────────────┘ │ ║ Bell Diameter:         0.40 units    ║  │
│ [Load Pasted]      │ ╚══════════════════════════════════════╝  │
│                    │                                            │
│ [Copy Current]     │ [View in Gallery] [Run Simulation]        │
│                    │                                            │
├────────────────────┴────────────────────────────────────────────┤
│ [Editor] [Gallery] [Education] [Compare]                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

**Phase 1: Basic Genome Editor (2-3 days)**
- Flask app setup with routing
- HTML/CSS for genome editor with 9 sliders
- API endpoint for server-side rendering using `make_jelly.py`
- Display rendered morphology image
- Copy/paste genome functionality

**Phase 2: Statistics & Validation (1 day)**
- Display phenotype stats from `fill_tank()`
- Genome validation (bounds checking)
- Error messages for invalid genomes
- Reset and randomize buttons

**Phase 3: Examples Gallery (1-2 days)**
- Load `best_genomes.json` and display evolution history
- Aurelia reference genome showcase
- Click to load in editor
- Thumbnail grid layout

**Phase 4: Educational Content (2-3 days)**
- Static HTML pages explaining genotype-phenotype mapping
- Diagrams (can use matplotlib to generate initially)
- Interactive Bezier curve demo (JavaScript Canvas API)
- Step-by-step morphology generation visualization

**Phase 5: Polish & Advanced Features (2-3 days)**
- Compare mode (split-screen)
- Improved styling (CSS gradients, animations)
- WebSocket for real-time updates (optional)
- Export functionality (JSON, PNG)
- Simulation preview (stretch goal)

**Total Estimate: 8-14 days**

---

## Constraints

- Must work on Termux (mobile browser)
- Server-side rendering initially (client-side optimization later)
- Scientific accuracy is critical (don't sacrifice correctness for aesthetics)
- Network latency: 200-500ms for server-side renders

---

## Dependencies

Add to `pyproject.toml`:
```toml
[project]
dependencies = [
  # ... existing deps
  "flask>=3.0.0",
  "pillow>=10.0.0",  # For image manipulation
  "flask-cors>=4.0.0",  # If frontend served separately
]
```

---

## Success Criteria

1. **Usability**: Researchers can explore morphology space within 5 minutes of first use
2. **Performance**: Morphology renders in <500ms
3. **Accuracy**: Rendered morphologies match simulation particle distributions
4. **Educational**: Students understand genotype-phenotype mapping after tutorial
5. **Accessibility**: Works on mobile browsers (Termux) and desktop
