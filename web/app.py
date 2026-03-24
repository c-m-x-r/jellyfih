"""
Minimalist genome viewer web interface.
Flask backend for jellyfish morphology visualization.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import csv
import subprocess
import json
import uuid
import re
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from make_jelly import fill_tank, random_genome, AURELIA_GENOME

app = Flask(__name__)

# Constants
N_PARTICLES = 80000
GRID_RES = 128
OUTPUT_DIR = Path(__file__).parent.parent / 'output'

# Custom submissions storage
CUSTOM_DIR = OUTPUT_DIR / 'custom_submissions'
CUSTOM_JSON = CUSTOM_DIR / 'submissions.json'
CUSTOM_THUMBS = CUSTOM_DIR / 'thumbnails'
CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_THUMBS.mkdir(parents=True, exist_ok=True)

# Genome bounds
GENOME_LOWER = [0.0, -0.15, 0.0, -0.2, 0.05, -0.30, 0.025, 0.025, 0.01]
GENOME_UPPER = [0.25, 0.15, 0.3, 0.15, 0.35, -0.03, 0.08, 0.1, 0.04]
GENOME_DEFAULT = [(lo + hi) / 2 for lo, hi in zip(GENOME_LOWER, GENOME_UPPER)]

# Material colors (minimalist palette)
MATERIAL_COLORS = {
    0: '#E8F4F8',  # Water: very light blue-gray
    1: '#4ECDC4',  # Jelly: teal
    2: '#FFA500',  # Payload: orange
    3: '#FF6B6B',  # Muscle: coral
}


def render_morphology(genome, size=(6, 8), colors=None):
    """Render morphology as base64 PNG. colors dict optionally overrides MATERIAL_COLORS by mat_id."""
    try:
        genome_arr = np.array(genome, dtype=np.float64)

        mat_colors = dict(MATERIAL_COLORS)
        if colors:
            mat_colors.update(colors)

        # Generate phenotype
        pos, mat, _, stats = fill_tank(genome_arr, N_PARTICLES, grid_res=GRID_RES)

        # Create figure with transparent background
        fig, ax = plt.subplots(figsize=size, facecolor='none')
        ax.set_facecolor('#FAFAFA')

        # Plot particles by material type
        for mat_id in [0, 1, 3, 2]:  # Water, jelly, muscle, payload (payload on top)
            mask = mat == mat_id
            if np.any(mask):
                ax.scatter(
                    pos[mask, 0], pos[mask, 1],
                    c=mat_colors[mat_id],
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


def parse_evolution_log(rel_path='evolution_log.csv'):
    """Parse an evolution log CSV and return list of dicts.
    rel_path is relative to OUTPUT_DIR (may include subdirectory).
    """
    # Security: resolve and ensure the path stays within OUTPUT_DIR
    csv_path = (OUTPUT_DIR / rel_path).resolve()
    if not str(csv_path).startswith(str(OUTPUT_DIR.resolve())):
        return []
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip duplicate header rows from resumed runs
            if row['generation'] == 'generation':
                continue
            rows.append({
                'generation': int(row['generation']),
                'individual': int(row['individual']),
                'genome': [float(row[f'gene_{i}']) for i in range(9)],
                'fitness': float(row['fitness']),
                'final_y': float(row['final_y']),
                'displacement': float(row['displacement']),
                'drift': float(row['drift']),
                'muscle_count': int(row['muscle_count']),
                'valid': int(row['valid']),
                'sigma': float(row['sigma'].strip()),
            })
    return rows


@app.route('/api/evolution/logs', methods=['GET'])
def api_evolution_logs():
    """List available evolution log files (searches recursively inside output/)."""
    logs = sorted(OUTPUT_DIR.rglob('*evolution_log*.csv'))
    result = []
    for p in logs:
        if ' ' in p.name:
            continue
        # Return path relative to OUTPUT_DIR so subdirectories are preserved
        result.append(str(p.relative_to(OUTPUT_DIR)))
    return jsonify({'logs': result})


@app.route('/api/evolution/summary', methods=['GET'])
def api_evolution_summary():
    """Return list of generations with best fitness per generation."""
    log_file = request.args.get('log', 'evolution_log.csv')
    rows = parse_evolution_log(log_file)
    if not rows:
        return jsonify({'generations': [], 'total_individuals': 0})

    gens = {}
    for r in rows:
        g = r['generation']
        if g not in gens:
            gens[g] = {'generation': g, 'count': 0, 'best_fitness': -1e9,
                       'avg_fitness': 0, 'valid_count': 0, 'sigma': r['sigma']}
        gens[g]['count'] += 1
        if r['valid']:
            gens[g]['avg_fitness'] += r['fitness']
            gens[g]['valid_count'] += 1
        if r['fitness'] > gens[g]['best_fitness']:
            gens[g]['best_fitness'] = r['fitness']

    gen_list = []
    for g in sorted(gens.keys()):
        info = gens[g]
        vc = info.pop('valid_count')
        info['avg_fitness'] = info['avg_fitness'] / vc if vc > 0 else info['best_fitness']
        gen_list.append(info)

    return jsonify({
        'generations': gen_list,
        'total_individuals': len(rows),
        'log': log_file
    })


@app.route('/api/evolution/generation/<int:gen>', methods=['GET'])
def api_evolution_generation(gen):
    """Return all individuals for a specific generation."""
    log_file = request.args.get('log', 'evolution_log.csv')
    rows = parse_evolution_log(log_file)
    individuals = [r for r in rows if r['generation'] == gen]
    individuals.sort(key=lambda r: r['fitness'], reverse=True)
    return jsonify({'generation': gen, 'individuals': individuals})


@app.route('/api/render/grid', methods=['POST'])
def api_render_grid():
    """Render up to 16 morphologies in a 4x4 grid."""
    data = request.get_json()
    individuals = data.get('individuals', [])

    if not individuals or len(individuals) > 16:
        return jsonify({'error': 'Provide 1-16 individuals'})

    n = len(individuals)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 5),
                             facecolor='#FAFAFA')
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        if idx < n:
            ind = individuals[idx]
            genome = ind['genome']
            try:
                pos, mat, _, stats = fill_tank(
                    np.array(genome, dtype=np.float64),
                    N_PARTICLES, grid_res=GRID_RES)

                for mat_id in [0, 1, 3, 2]:
                    mask = mat == mat_id
                    if np.any(mask):
                        ax.scatter(
                            pos[mask, 0], pos[mask, 1],
                            c=MATERIAL_COLORS[mat_id],
                            s=0.5, alpha=0.8 if mat_id == 0 else 1.0,
                            edgecolors='none', rasterized=True)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')

                label = f"#{ind.get('individual', idx)}"
                fitness = ind.get('fitness', None)
                if fitness is not None:
                    label += f"  f={fitness:.3f}"
                ax.set_title(label, fontsize=10, fontfamily='monospace',
                             color='#1A1A1A', pad=4)
            except Exception:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                        color='#FF6B6B', fontsize=12)

        ax.axis('off')

    plt.tight_layout(pad=1)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight',
                facecolor='#FAFAFA')
    plt.close(fig)

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({'image': f'data:image/png;base64,{img_base64}'})


PROJECT_DIR = Path(__file__).parent.parent

# Track running simulation subprocess
_sim_process = None


@app.route('/api/simulate/generation', methods=['POST'])
def api_simulate_generation():
    """Start a GPU simulation of a full generation. Returns immediately."""
    global _sim_process

    # Reject if already running
    if _sim_process is not None and _sim_process.poll() is None:
        return jsonify({'error': 'A simulation is already running'}), 409

    data = request.get_json()
    gen = data.get('generation')
    log_file = data.get('log', 'evolution_log.csv')
    n_frames = data.get('frames', 100)
    web_palette = data.get('web_palette', True)

    if gen is None:
        return jsonify({'error': 'generation is required'}), 400

    # Sanitize log filename
    log_file = Path(log_file).name

    # Use just the filename stem (no subdir) for the video name
    log_stem = Path(log_file).stem
    # Predictable run_id so we know exactly where the video will land
    sim_run_id = f"sim_{log_stem}_gen{gen}"

    cmd = [
        sys.executable, 'evolve.py',
        '--sim-gen', '--gen', str(gen),
        '--log', log_file,
        '--frames', str(n_frames),
        '--run-id', sim_run_id,
    ]
    if web_palette:
        cmd.append('--web-palette')

    # Clear any old status file
    status_path = OUTPUT_DIR / 'sim_status.json'
    if status_path.exists():
        status_path.unlink()

    _sim_process = subprocess.Popen(
        cmd, cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )

    # Video lands at output/{sim_run_id}/sim_{log_stem}_gen{gen}.mp4
    expected_video = f"{sim_run_id}/sim_{log_stem}_gen{gen}.mp4"

    return jsonify({
        'status': 'started',
        'video': expected_video,
        'generation': gen,
        'total_frames': n_frames,
    })


@app.route('/api/simulate/status', methods=['GET'])
def api_simulate_status():
    """Poll simulation progress from the status JSON file."""
    global _sim_process

    status_path = OUTPUT_DIR / 'sim_status.json'

    # Check if process has exited
    proc_running = _sim_process is not None and _sim_process.poll() is None

    if status_path.exists():
        try:
            with open(status_path, 'r') as f:
                status = json.load(f)
            # If process finished but status still says running, mark done or error
            if not proc_running and status.get('state') == 'running':
                status['state'] = 'error'
                status['error'] = 'Process exited unexpectedly'
            return jsonify(status)
        except (json.JSONDecodeError, IOError):
            pass

    if proc_running:
        return jsonify({'state': 'starting', 'frame': 0, 'total_frames': 0})

    return jsonify({'state': 'idle'})


@app.route('/api/simulate/video/<path:filename>')
def api_simulate_video(filename):
    """Serve a simulation video file (supports subdirectory paths within output/)."""
    # Security: ensure resolved path stays within OUTPUT_DIR
    target = (OUTPUT_DIR / filename).resolve()
    if not str(target).startswith(str(OUTPUT_DIR.resolve())):
        return 'Forbidden', 403
    return send_from_directory(str(OUTPUT_DIR), filename, mimetype='video/mp4')


@app.route('/custom')
def custom():
    """Serve the custom jellyfish design page."""
    return render_template('custom.html')


@app.route('/api/custom/render', methods=['POST'])
def api_custom_render():
    """Render with custom jelly/muscle colors for the design wizard."""
    data = request.get_json()
    genome = data.get('genome', GENOME_DEFAULT)
    jelly_color = data.get('jelly_color', MATERIAL_COLORS[1])
    muscle_color = data.get('muscle_color', MATERIAL_COLORS[3])
    result = render_morphology(genome, colors={1: jelly_color, 3: muscle_color})
    return jsonify(result)


def _load_submissions():
    if not CUSTOM_JSON.exists():
        return []
    try:
        with open(CUSTOM_JSON, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_submissions(submissions):
    with open(CUSTOM_JSON, 'w') as f:
        json.dump(submissions, f, indent=2)


def predict_performance(genome):
    """Find nearest neighbour in evolution logs, return percentile + displacement estimate."""
    logs = sorted(OUTPUT_DIR.rglob('*evolution_log*.csv'))
    # Exclude custom-related paths
    logs = [l for l in logs if 'custom' not in str(l)]
    if not logs:
        return None
    rows = parse_evolution_log(str(logs[-1].relative_to(OUTPUT_DIR)))
    valid_rows = [r for r in rows if r['valid']]
    if not valid_rows:
        return None

    ranges = np.array(GENOME_UPPER) - np.array(GENOME_LOWER)
    query_norm = (np.array(genome) - np.array(GENOME_LOWER)) / ranges
    genomes_norm = (np.array([r['genome'] for r in valid_rows]) - np.array(GENOME_LOWER)) / ranges
    distances = np.linalg.norm(genomes_norm - query_norm, axis=1)
    nearest = valid_rows[int(np.argmin(distances))]

    all_fitness = [r['fitness'] for r in valid_rows]
    percentile = float(100 * np.mean(np.array(all_fitness) <= nearest['fitness']))

    return {
        'nearest_fitness': nearest['fitness'],
        'displacement': nearest['displacement'],
        'generation': nearest['generation'],
        'percentile': round(percentile, 1),
        'total_individuals': len(valid_rows),
    }


@app.route('/api/custom/submit', methods=['POST'])
def api_custom_submit():
    """Accept a user-designed jellyfish submission."""
    data = request.get_json()

    name = (data.get('name') or '').strip()[:60]
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    genome = data.get('genome')
    if not genome or len(genome) != 9:
        return jsonify({'error': 'Invalid genome'}), 400
    for i, v in enumerate(genome):
        if v < GENOME_LOWER[i] or v > GENOME_UPPER[i]:
            return jsonify({'error': f'Gene {i} out of bounds'}), 400

    jelly_color = data.get('jelly_color', MATERIAL_COLORS[1])
    muscle_color = data.get('muscle_color', MATERIAL_COLORS[3])

    # Validate hex colors (basic check)
    hex_re = re.compile(r'^#[0-9a-fA-F]{6}$')
    if not hex_re.match(jelly_color) or not hex_re.match(muscle_color):
        return jsonify({'error': 'Invalid color format'}), 400

    email = (data.get('email') or '').strip()[:254] or None

    submission_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Render and save thumbnail
    try:
        thumb_result = render_morphology(
            genome, size=(4, 5),
            colors={1: jelly_color, 3: muscle_color}
        )
        if 'image' in thumb_result:
            img_data = thumb_result['image'].split(',', 1)[1]
            thumb_path = CUSTOM_THUMBS / f'{submission_id}.png'
            with open(thumb_path, 'wb') as f:
                f.write(base64.b64decode(img_data))
    except Exception:
        pass  # Thumbnail failure is non-fatal

    submission = {
        'id': submission_id,
        'name': name,
        'genome': genome,
        'jelly_color': jelly_color,
        'muscle_color': muscle_color,
        'timestamp': timestamp,
    }
    if email:
        submission['email'] = email

    submissions = _load_submissions()
    submissions.append(submission)
    _save_submissions(submissions)

    prediction = predict_performance(genome)

    return jsonify({'id': submission_id, 'prediction': prediction})


@app.route('/api/custom/aquarium', methods=['GET'])
def api_custom_aquarium():
    """Return the last ≤15 submissions for the aquarium display."""
    submissions = _load_submissions()
    recent = submissions[-16:][::-1]  # Most recent first, up to 16 (2 cols × 8)
    public = [
        {
            'id': s['id'],
            'name': s['name'],
            'jelly_color': s['jelly_color'],
            'muscle_color': s['muscle_color'],
            'timestamp': s['timestamp'],
        }
        for s in recent
    ]
    return jsonify({'submissions': public})


@app.route('/api/custom/thumbnail/<submission_id>')
def api_custom_thumbnail(submission_id):
    """Serve a submission thumbnail PNG."""
    # Only allow UUID-format IDs
    if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', submission_id):
        return 'Not found', 404
    thumb_path = CUSTOM_THUMBS / f'{submission_id}.png'
    if not thumb_path.exists():
        return 'Not found', 404
    return send_from_directory(str(CUSTOM_THUMBS), f'{submission_id}.png', mimetype='image/png')


if __name__ == '__main__':
    # Run on all interfaces to allow Termux access
    app.run(host='0.0.0.0', port=5000, debug=True)
