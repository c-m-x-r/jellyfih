"""
Minimalist genome viewer web interface.
Flask backend for jellyfish morphology visualization.
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from make_jelly import fill_tank, random_genome, AURELIA_GENOME

app = Flask(__name__)

# Constants
N_PARTICLES = 80000
GRID_RES = 128

# Genome bounds
GENOME_LOWER = [0.0, -0.15, 0.0, -0.2, 0.05, -0.45, 0.025, 0.025, 0.01]
GENOME_UPPER = [0.25, 0.15, 0.3, 0.15, 0.35, -0.03, 0.08, 0.1, 0.04]
GENOME_DEFAULT = [(lo + hi) / 2 for lo, hi in zip(GENOME_LOWER, GENOME_UPPER)]

# Material colors (minimalist palette)
MATERIAL_COLORS = {
    0: '#E8F4F8',  # Water: very light blue-gray
    1: '#4ECDC4',  # Jelly: teal
    2: '#FFA500',  # Payload: orange
    3: '#FF6B6B',  # Muscle: coral
}


def render_morphology(genome, size=(6, 8)):
    """Render morphology as base64 PNG."""
    try:
        genome_arr = np.array(genome, dtype=np.float64)

        # Generate phenotype
        pos, mat, stats = fill_tank(genome_arr, N_PARTICLES, grid_res=GRID_RES)

        # Create figure with transparent background
        fig, ax = plt.subplots(figsize=size, facecolor='none')
        ax.set_facecolor('#FAFAFA')

        # Plot particles by material type
        for mat_id in [0, 1, 3, 2]:  # Water, jelly, muscle, payload (payload on top)
            mask = mat == mat_id
            if np.any(mask):
                ax.scatter(
                    pos[mask, 0], pos[mask, 1],
                    c=MATERIAL_COLORS[mat_id],
                    s=1.0,
                    alpha=0.8 if mat_id == 0 else 1.0,
                    edgecolors='none',
                    rasterized=True
                )

        # Styling
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Save to bytes
        buf = io.BytesIO()
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    pad_inches=0, facecolor='none', transparent=False)
        plt.close(fig)

        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return {
            'image': f'data:image/png;base64,{img_base64}',
            'stats': stats
        }
    except Exception as e:
        return {'error': str(e)}


@app.route('/')
def index():
    """Serve the main viewer page."""
    return render_template('index.html')


@app.route('/api/render', methods=['POST'])
def api_render():
    """Render a genome and return image + stats."""
    data = request.get_json()
    genome = data.get('genome', GENOME_DEFAULT)

    result = render_morphology(genome)
    return jsonify(result)


@app.route('/api/random', methods=['GET'])
def api_random():
    """Generate a random genome."""
    genome = random_genome()
    return jsonify({'genome': genome.tolist()})


@app.route('/api/aurelia', methods=['GET'])
def api_aurelia():
    """Get the Aurelia reference genome."""
    return jsonify({'genome': AURELIA_GENOME.tolist()})


@app.route('/api/default', methods=['GET'])
def api_default():
    """Get the default genome."""
    return jsonify({'genome': GENOME_DEFAULT})


@app.route('/api/bounds', methods=['GET'])
def api_bounds():
    """Get genome parameter bounds."""
    return jsonify({
        'lower': GENOME_LOWER,
        'upper': GENOME_UPPER,
        'default': GENOME_DEFAULT
    })


if __name__ == '__main__':
    # Run on all interfaces to allow Termux access
    app.run(host='0.0.0.0', port=5000, debug=True)
