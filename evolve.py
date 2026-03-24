"""
Evolutionary optimization of jellyfish morphologies using CMA-ES.

Evaluates 16 morphologies in parallel on GPU via MPM simulation.
Fitness: efficiency — vertical displacement per unit muscle investment (sqrt-normalised).

The payload is slightly negatively buoyant (2.5x density, 0.44x gravity = 1.1x
effective weight). Without active swimming, the payload sinks. Fitness rewards
morphologies that maintain the highest payload altitude with lateral stability.

Usage:
    uv run python evolve.py                  # Full 50-generation run
    uv run python evolve.py --gens 5         # Quick 5-generation test
    uv run python evolve.py --view           # Render best genomes as video
    uv run python evolve.py --view --gen 3   # Render best from generation 3
"""

import argparse
import numpy as np
import cma
import time
import os
import csv
import json
import pickle

# Pre-parse --instances so env var is set before mpm_sim import (Taichi allocates on import)
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--instances', type=int, default=16)
_pre_args, _ = _pre.parse_known_args()
os.environ['JELLY_INSTANCES'] = str(_pre_args.instances)

import taichi as ti
import mpm_sim as sim
from make_jelly import fill_tank, random_genome, AURELIA_GENOME

# --- CONFIGURATION ---
GENERATIONS = 50
STEPS_PER_EVAL = 150000  # 3 actuation cycles at 1Hz, dt=2e-5
POPSIZE = sim.n_instances  # Must match GPU instances (16)
START_SIGMA = 0.25
OUTPUT_DIR = "output"

# Genome bounds: [cp1_x, cp1_y, cp2_x, cp2_y, end_x, end_y, t_base, t_mid, t_tip]
GENOME_LOWER = [0.0, -0.15,  0.0,  -0.2,  0.05, -0.30,  0.025,  0.025,  0.01]
GENOME_UPPER = [0.25,  0.15,  0.3,   0.15, 0.35, -0.03,  0.08,   0.1,    0.04]
GENOME_X0 = [(lo + hi) / 2 for lo, hi in zip(GENOME_LOWER, GENOME_UPPER)]

# Fitness constants
PENALTY_INVALID = 100.0
MUSCLE_FLOOR = 200        # Below this → degenerate morphology, treated as invalid
MUSCLE_REFERENCE = 500    # Median expected muscle count; efficiency baseline

# View mode constants
VIEW_STEPS = 150000      # Sim steps for rendered view (3 actuation cycles at 1Hz, dt=2e-5)
VIEW_RENDER_EVERY = 500  # Render a frame every N substeps (~same sim-time per frame as before)
VIEW_FPS = 30

from mpm_sim import WEB_PALETTE  # centralised in mpm_sim.py


def render_frame(radius, web_palette=False):
    """Render one frame using either the abyss or web palette pipeline."""
    if web_palette:
        sim.clear_frame_buffer_white()
        for mat_id, r, g, b in WEB_PALETTE:
            sim.render_flat_pass(sim.res_sub, sim.grid_side, radius, mat_id, r, g, b)
    else:
        sim.clear_frame_buffer()
        sim.render_frame_abyss(sim.res_sub, sim.grid_side, radius)
        sim.tone_map_and_encode()


def compute_fitness(sim_results, muscle_counts):
    raw = []
    for i in range(POPSIZE):
        init_y = sim_results[i, 0]
        final_y = sim_results[i, 2]
        valid = sim_results[i, 4]

        if valid == 0 or muscle_counts[i] < MUSCLE_FLOOR:
            raw.append(None)
            continue

        # Cap at ceiling so bouncing doesn't inflate score
        displacement = min(final_y, 0.93) - init_y
        muscle_cost = (muscle_counts[i] / MUSCLE_REFERENCE) ** 0.5
        raw.append(-displacement / muscle_cost)  # CMA-ES minimises

    # Set penalty to just worse than the worst valid individual this generation.
    # Avoids a massive magnitude gap that distorts CMA-ES covariance updates.
    valid_scores = [s for s in raw if s is not None]
    penalty = (max(valid_scores) + 1.0) if valid_scores else PENALTY_INVALID

    return [penalty if s is None else s for s in raw]


def load_batch(genomes):
    """
    Generate phenotypes and load all instances to GPU.
    Returns muscle_counts list and per-instance stats.
    """
    muscle_counts = []
    instance_stats = []

    for i, genome in enumerate(genomes):
        pos, mat, fiber, stats = fill_tank(
            genome, sim.n_particles, grid_res=int(sim.n_grid)
        )
        sim.load_particles(i, pos, mat, fiber)
        muscle_counts.append(stats['muscle_count'])
        instance_stats.append(stats)

    ti.sync()
    return muscle_counts, instance_stats


def run_baseline():
    """
    Zero-actuation baseline: load the centre genome with actuation disabled.
    Checks that a passive jellyfish body doesn't drift significantly.
    """
    print("Running zero-actuation baseline (centre genome, actuation=0)...")

    genome = GENOME_X0
    genomes = [genome] * POPSIZE
    load_batch(genomes)

    # Disable actuation for all instances
    saved = [sim.instance_actuation[m] for m in range(POPSIZE)]
    for m in range(POPSIZE):
        sim.instance_actuation[m] = 0.0

    results = sim.run_batch_headless(20000)

    # Restore actuation
    for m in range(POPSIZE):
        sim.instance_actuation[m] = saved[m]

    displacement = results[0, 2] - results[0, 0]
    if abs(displacement) > 0.05:
        print(f"  WARNING: Passive body drifted {displacement:+.4f} with zero actuation "
              f"— gravity/buoyancy artefact?")
    else:
        print(f"  OK: Zero-actuation drift = {displacement:+.4f}")
    return displacement


def run_payload_sink_baseline():
    """
    Payload-only baseline: no jellyfish, just payload + water.
    Verifies the payload genuinely sinks under gravity before evolution starts.
    Runs all instances (substep is batch) but only inspects instance 0.
    """
    from scipy.spatial import cKDTree as _cKDTree
    from make_jelly import PAYLOAD_WIDTH, PAYLOAD_HEIGHT, DEFAULT_SPAWN

    print("Running payload-sink baseline (no jellyfish, centered at 0.5)...")

    spacing = 1.0 / (sim.n_grid * 2.0)
    margin = spacing * 3
    raster_res = sim.n_grid * 2

    # Drop payload at the true centre of the domain so buoyancy/gravity are clear
    center_spawn = np.array([0.5, 0.5])

    px = np.linspace(-PAYLOAD_WIDTH / 2, PAYLOAD_WIDTH / 2,
                     int(PAYLOAD_WIDTH * raster_res))
    py = np.linspace(0, PAYLOAD_HEIGHT, int(PAYLOAD_HEIGHT * raster_res))
    pgx, pgy = np.meshgrid(px, py)
    payload_pos = np.vstack([pgx.ravel(), pgy.ravel()]).T + center_spawn
    n_payload = len(payload_pos)

    wx = np.arange(margin, 1.0 - margin, spacing)
    wy = np.arange(margin, 1.0 - margin, spacing)
    wgx, wgy = np.meshgrid(wx, wy)
    water_candidates = np.vstack([wgx.ravel(), wgy.ravel()]).T
    dists, _ = _cKDTree(payload_pos).query(water_candidates, k=1)
    water_pos = water_candidates[dists > 0.005]
    n_water = min(len(water_pos), sim.n_particles - n_payload)

    positions = np.full((sim.n_particles, 2), -1.0, dtype=np.float32)
    materials = np.full(sim.n_particles, -1, dtype=np.int32)
    fibers = np.zeros((sim.n_particles, 2), dtype=np.float32)
    fibers[:, 1] = 1.0
    positions[:n_payload] = payload_pos
    materials[:n_payload] = 2
    positions[n_payload:n_payload + n_water] = water_pos[:n_water]
    materials[n_payload:n_payload + n_water] = 0

    for m in range(POPSIZE):
        sim.load_particles(m, positions, materials, fibers)
    ti.sync()

    results = sim.run_batch_headless(20000)
    displacement = results[0, 2] - results[0, 0]

    if displacement > 0.005:
        print(f"  WARNING: Payload ROSE {displacement:+.4f} without jellyfish — "
              f"check gravity/density settings.")
    elif displacement < -0.005:
        print(f"  OK: Payload sank {abs(displacement):.4f} as expected.")
    else:
        print(f"  NOTE: Payload barely moved ({displacement:+.4f}) — nearly neutral.")
    return displacement


def save_checkpoint(es, generation, history, filepath):
    """Save CMA-ES state for crash recovery."""
    checkpoint = {
        'generation': generation,
        'history': history,
        'cma_state': es.pickle_dumps(),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(filepath):
    """Load CMA-ES state from checkpoint. Returns (es, generation, history) or None."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        es = pickle.loads(checkpoint['cma_state'])
        print(f"Resumed from checkpoint at generation {checkpoint['generation']}")
        return es, checkpoint['generation'], checkpoint['history']
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None


def view_best(gen_idx=None, web_palette=False):
    """
    Load best genomes from evolution results and render them in the 4x4 grid.
    Layout: 4 rows (generations) x 4 columns (same genome, color-coded).
    Column colors: lime, green, turquoise, cyan (abyss) or flat web palette.
    Saves an MP4 video to output/.
    """
    import imageio.v3 as iio

    json_path = os.path.join(OUTPUT_DIR, "best_genomes.json")
    if not os.path.exists(json_path):
        print(f"No results found at {json_path}. Run evolution first.")
        return

    with open(json_path) as f:
        history = json.load(f)

    if not history:
        print("No generations in history.")
        return

    # Column hues: lime, green, turquoise, cyan
    col_hues = [0.22, 0.33, 0.44, 0.50]
    grid_side = sim.grid_side  # 4

    # Pick 4 generations for the 4 rows
    if gen_idx is not None:
        # Load all individuals from the full CSV log for this generation.
        # Each GPU instance shows a different individual (sorted by fitness).
        # Falls back to best_genomes.json single entry if no CSV is found.
        csv_path = os.path.join(OUTPUT_DIR, "evolution_log.csv")
        run_id = os.path.basename(os.path.abspath(OUTPUT_DIR))
        alt_csv = os.path.join(OUTPUT_DIR, f"evolution_log_{run_id}.csv")
        if not os.path.exists(csv_path) and os.path.exists(alt_csv):
            csv_path = alt_csv

        use_genomes = []
        sorted_ind = []
        if os.path.exists(csv_path):
            individuals = load_generation_from_csv(csv_path, gen_idx)
            sorted_ind = sorted(individuals, key=lambda r: r['fitness'], reverse=True)
            use_genomes = [np.array(r['genome']) for r in sorted_ind[:POPSIZE]]

        if not use_genomes:
            # Fallback: single best genome from best_genomes.json
            entry = next((h for h in history if h['generation'] == gen_idx), None)
            if entry is None:
                print(f"Generation {gen_idx} not found in history or CSV.")
                return
            use_genomes = [np.array(entry['genome'])]

        # Pad to POPSIZE with last entry
        while len(use_genomes) < POPSIZE:
            use_genomes.append(use_genomes[-1])
        genomes = use_genomes
        label = f"gen_{gen_idx}"
        n_unique = min(len(sorted_ind), POPSIZE) if sorted_ind else 1
        fitness_str = (f"fitness range: {sorted_ind[0]['fitness']:.4f} → "
                       f"{sorted_ind[min(n_unique-1, len(sorted_ind)-1)]['fitness']:.4f}"
                       if sorted_ind else "from best_genomes.json")
        print(f"Viewing generation {gen_idx}: {n_unique} distinct individuals ({fitness_str})")
        print(f"  Layout: each cell = different individual, ranked by fitness")
    else:
        # Pick up to 4 evenly spaced generations; each row = one generation's best
        n_avail = len(history)
        if n_avail >= grid_side:
            indices = np.linspace(0, n_avail - 1, grid_side, dtype=int)
        else:
            indices = list(range(n_avail))
        row_entries = [history[i] for i in indices]
        while len(row_entries) < grid_side:
            row_entries.append(row_entries[-1])
        label = "progression"
        print(f"Viewing progression: rows = gen "
              + ", ".join(str(e['generation']) for e in row_entries))
        print(f"  Columns: lime | green | turquoise | cyan")

        # Build the 16-instance genome list: row=generation, col=color variant
        genomes = []
        for row in range(grid_side):
            for col in range(grid_side):
                genomes.append(row_entries[row]['genome'])

    # Set per-instance hues by column (only used for abyss palette)
    if not web_palette:
        for m in range(POPSIZE):
            col = m % grid_side
            sim.instance_hue[m] = col_hues[col]

    # Load genomes into GPU
    print("Loading phenotypes...", flush=True)
    load_batch(genomes)

    # Run simulation with rendering
    palette_suffix = "_web" if web_palette else ""
    video_path = os.path.join(OUTPUT_DIR, f"view_{label}{palette_suffix}.mp4")
    total_frames = VIEW_STEPS // VIEW_RENDER_EVERY
    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)

    palette_name = "web" if web_palette else "abyss"
    print(f"Rendering {total_frames} frames ({VIEW_STEPS} steps, "
          f"1 frame every {VIEW_RENDER_EVERY} substeps, palette: {palette_name})...")

    sim.sim_time[None] = 0.0
    frames = []

    for step in range(VIEW_STEPS):
        sim.substep()

        if step % VIEW_RENDER_EVERY == 0:
            render_frame(radius, web_palette=web_palette)
            ti.sync()

            img = sim.frame_buffer.to_numpy()
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            frames.append(img_uint8)

            frame_num = step // VIEW_RENDER_EVERY
            if frame_num % 50 == 0:
                print(f"  Frame {frame_num}/{total_frames} "
                      f"(t={sim.sim_time[None]:.3f}s)", flush=True)

    print(f"Writing video to {video_path}...")
    iio.imwrite(video_path, frames, fps=VIEW_FPS)
    print(f"Done! {len(frames)} frames written to {video_path}")


def evolve(generations, seed=42):
    """Run the evolutionary loop."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_id = os.path.basename(os.path.abspath(OUTPUT_DIR))
    checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_{run_id}.pkl")
    csv_path = os.path.join(OUTPUT_DIR, f"evolution_log_{run_id}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"best_genomes_{run_id}.json")

    # Try to resume from checkpoint
    resumed = load_checkpoint(checkpoint_path)
    if resumed:
        es, start_gen, history = resumed
        start_gen += 1  # Continue from next generation
    else:
        # Sanity checks before evolution starts
        run_payload_sink_baseline()  # payload sinks without jellyfish
        run_baseline()               # passive body doesn't drift with zero actuation

        # Initialize CMA-ES with built-in bounds handling
        # CMA_stds normalizes the search space so mutations are proportional
        # to each gene's range (e.g. t_tip range 0.03 vs end_y range 0.42)
        gene_ranges = [hi - lo for lo, hi in zip(GENOME_LOWER, GENOME_UPPER)]
        opts = {
            'popsize': POPSIZE,
            'bounds': [GENOME_LOWER, GENOME_UPPER],
            'seed': seed,
            'maxiter': generations,
            'CMA_stds': gene_ranges,
        }
        es = cma.CMAEvolutionStrategy(GENOME_X0, START_SIGMA, opts)
        start_gen = 0
        history = []

    # Prepare CSV (append mode for resume)
    csv_exists = os.path.exists(csv_path) and start_gen > 0
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        header = ['generation', 'individual'] + [f'gene_{i}' for i in range(9)] + \
                 ['fitness', 'final_y', 'displacement', 'drift', 'muscle_count', 'valid', 'sigma', 'efficiency']
        csv_writer.writerow(header)
        csv_file.flush()

    print(f"\nStarting evolution: popsize={POPSIZE}, generations={generations}, "
          f"steps/eval={STEPS_PER_EVAL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)

    n_invalid_total = 0

    for gen in range(start_gen, generations):
        if es.stop():
            print(f"CMA-ES stop condition reached at generation {gen}")
            break

        # 1. Get candidate genomes from CMA-ES
        genomes = es.ask()

        # 2. Generate phenotypes and load to GPU (CPU)
        t0 = time.time()
        muscle_counts, batch_stats = load_batch(genomes)
        t_load = time.time() - t0

        # 3. Run physics simulation (GPU, headless)
        t0 = time.time()
        sim_results = sim.run_batch_headless(STEPS_PER_EVAL)
        t_sim = time.time() - t0

        # 3.5. Mark self-intersecting morphologies as invalid
        n_self_intersect = 0
        for i in range(POPSIZE):
            if batch_stats[i]['self_intersecting']:
                sim_results[i, 4] = 0.0
                n_self_intersect += 1

        # 4. Compute fitness
        fitness_values = compute_fitness(sim_results, muscle_counts)

        # 5. Update CMA-ES
        es.tell(genomes, fitness_values)

        # 6. Logging
        best_idx = np.argmin(fitness_values)
        best_fitness = -fitness_values[best_idx]
        best_disp = sim_results[best_idx, 2] - sim_results[best_idx, 0]
        n_invalid = sum(1 for f in fitness_values if f > 0)  # positive = penalty
        valid_scores = [-f for f in fitness_values if f <= 0]
        avg_fitness = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        n_invalid_total += n_invalid

        # Per-individual CSV logging
        for ind in range(POPSIZE):
            disp = sim_results[ind, 2] - sim_results[ind, 0]
            drift = abs(sim_results[ind, 3] - sim_results[ind, 1])
            valid = sim_results[ind, 4]
            final_y = sim_results[ind, 2]
            mc = muscle_counts[ind]
            eff = (disp / ((mc / MUSCLE_REFERENCE) ** 0.5)) if (mc >= MUSCLE_FLOOR and valid) else 0.0
            row = [gen, ind] + list(genomes[ind]) + [
                -fitness_values[ind], final_y, disp, drift,
                mc, int(valid), es.sigma, eff
            ]
            csv_writer.writerow(row)
        csv_file.flush()

        # Covariance diagnostics: eigenvalue spread and top correlations
        C_arr = np.array(es.sm.C)
        eigvals = np.linalg.eigvalsh(C_arr)
        cov_diag = C_arr.diagonal().tolist()

        # Best genome per generation
        history.append({
            'generation': gen,
            'genome': genomes[best_idx].tolist(),
            'fitness': float(best_fitness),
            'avg_fitness': float(avg_fitness),
            'displacement': float(best_disp),
            'sigma': float(es.sigma),
            'cov_cond': float(eigvals[-1] / max(eigvals[0], 1e-12)),
            'cov_diag': [round(float(v), 4) for v in cov_diag],
        })
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Console output
        best_alt = sim_results[best_idx, 2]
        cond = float(eigvals[-1] / max(eigvals[0], 1e-12))
        si_str = f" SI: {n_self_intersect}" if n_self_intersect > 0 else ""
        print(f"Gen {gen:3d} | Best: {best_fitness:+.4f} Avg: {avg_fitness:+.4f} | "
              f"Alt: {best_alt:.4f} | "
              f"Disp: {best_disp:+.4f} | "
              f"Sigma: {es.sigma:.4f} | Cond: {cond:.1f} | "
              f"Invalid: {n_invalid}/{POPSIZE}{si_str} | "
              f"Load: {t_load:.1f}s Sim: {t_sim:.1f}s")

        # Warn if too many invalids
        if n_invalid > POPSIZE * 0.5:
            print(f"  WARNING: {n_invalid}/{POPSIZE} invalid evaluations this generation")

        # Checkpoint every 5 generations
        if gen % 5 == 0 or gen == generations - 1:
            save_checkpoint(es, gen, history, checkpoint_path)

    csv_file.close()

    # Final summary
    print("\n" + "=" * 60)
    print("Evolution complete.")
    if history:
        best_ever = max(history, key=lambda h: h['fitness'])
        print(f"Best fitness: {best_ever['fitness']:.4f} at generation {best_ever['generation']}")
        print(f"Best genome: {best_ever['genome']}")
        print(f"Total invalid evaluations: {n_invalid_total}")
    print(f"Results saved to {OUTPUT_DIR}/")


def eval_aurelia(web_palette=False):
    """
    Evaluate the Aurelia aurita reference genome and render a video.
    Runs the same simulation as evolution but with the hand-designed
    moon jelly morphology, reporting its fitness as a baseline.
    """
    import imageio.v3 as iio

    print("Evaluating Aurelia aurita (moon jelly) reference genome...")
    print(f"  Genome: {AURELIA_GENOME.tolist()}")

    # Fill all 16 instances with the same Aurelia genome
    genomes = [AURELIA_GENOME] * POPSIZE
    muscle_counts, batch_stats = load_batch(genomes)

    print(f"  Muscle count: {muscle_counts[0]}")
    print(f"  Robot particles: {batch_stats[0]['n_robot']}")

    # Run headless evaluation
    print(f"  Running {STEPS_PER_EVAL} simulation steps...")
    sim_results = sim.run_batch_headless(STEPS_PER_EVAL)

    # Compute fitness
    fitness_values = compute_fitness(sim_results, muscle_counts)
    best_fitness = -fitness_values[0]
    displacement = sim_results[0, 2] - sim_results[0, 0]
    drift = abs(sim_results[0, 3] - sim_results[0, 1])

    print(f"\n  Results:")
    print(f"    Fitness (staying power): {best_fitness:.4f}")
    print(f"    Final altitude:          {sim_results[0, 2]:.4f}")
    print(f"    Displacement:            {displacement:+.4f}")
    print(f"    Lateral drift:           {drift:.4f}")
    print(f"    Valid:                    {sim_results[0, 4]}")

    # Render video with all 16 showing the same Aurelia
    if not web_palette:
        col_hues = [0.22, 0.33, 0.44, 0.50]
        for m in range(POPSIZE):
            col = m % sim.grid_side
            sim.instance_hue[m] = col_hues[col]

    # Reload and re-simulate with rendering
    load_batch(genomes)
    palette_suffix = "_web" if web_palette else ""
    video_path = os.path.join(OUTPUT_DIR, f"view_aurelia{palette_suffix}.mp4")
    total_frames = VIEW_STEPS // VIEW_RENDER_EVERY
    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)

    palette_name = "web" if web_palette else "abyss"
    print(f"\n  Rendering {total_frames} frames to {video_path} (palette: {palette_name})...")
    sim.sim_time[None] = 0.0
    frames = []

    for step in range(VIEW_STEPS):
        sim.substep()
        if step % VIEW_RENDER_EVERY == 0:
            render_frame(radius, web_palette=web_palette)
            ti.sync()
            img = sim.frame_buffer.to_numpy()
            frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))

    iio.imwrite(video_path, frames, fps=VIEW_FPS)
    print(f"  Done! Video saved to {video_path}")


def load_generation_from_csv(csv_path, gen):
    """Load all individuals for a generation from an evolution log CSV."""
    individuals = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['generation'] == 'generation':
                continue  # Skip duplicate headers
            if int(row['generation']) == gen:
                genome = [float(row[f'gene_{i}']) for i in range(9)]
                individuals.append({
                    'individual': int(row['individual']),
                    'genome': genome,
                    'fitness': float(row['fitness']),
                })
    individuals.sort(key=lambda r: r['individual'])
    return individuals


def sim_generation(gen, log_file='evolution_log.csv', n_frames=100,
                   web_palette=False):
    """
    Simulate all individuals from a generation and render as video.
    Reads genomes from a CSV log file, loads all 16 into the GPU,
    and renders n_frames of MPM simulation as a 4x4 grid MP4.
    Writes progress to output/sim_status.json for web frontend polling.
    """
    import imageio.v3 as iio

    # log_file may be relative to base output/ dir (e.g. "run_id/evolution_log_run_id.csv")
    base_output = os.path.dirname(os.path.abspath(OUTPUT_DIR))
    csv_path = os.path.join(base_output, log_file)
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None

    individuals = load_generation_from_csv(csv_path, gen)
    if not individuals:
        print(f"Error: No individuals found for generation {gen} in {log_file}")
        return None

    n_ind = len(individuals)
    print(f"Simulating generation {gen} from {log_file}: {n_ind} individuals, "
          f"{n_frames} frames")

    # Build genome list for all 16 GPU slots
    genomes = []
    for i in range(POPSIZE):
        if i < n_ind:
            genomes.append(np.array(individuals[i]['genome']))
        else:
            # Pad with the last individual if fewer than 16
            genomes.append(np.array(individuals[-1]['genome']))

    # Set hues (not used for web palette, but harmless)
    col_hues = [0.22, 0.33, 0.44, 0.50]
    for m in range(POPSIZE):
        col = m % sim.grid_side
        sim.instance_hue[m] = col_hues[col]

    # Load phenotypes
    print("Loading phenotypes...", flush=True)
    load_batch(genomes)

    # Rendering setup — use just the filename stem (strip any subdirectory from log_file)
    log_stem = os.path.splitext(os.path.basename(log_file))[0]
    video_name = f"sim_{log_stem}_gen{gen}.mp4"
    video_path = os.path.join(OUTPUT_DIR, video_name)
    # Status file goes to base output/ dir so Flask can poll it regardless of run subdir
    status_path = os.path.join(base_output, "sim_status.json")

    total_steps = n_frames * VIEW_RENDER_EVERY
    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)
    palette_name = "web" if web_palette else "abyss"

    print(f"Rendering {n_frames} frames ({total_steps} steps, "
          f"palette: {palette_name})...")

    sim.sim_time[None] = 0.0
    frames = []

    # Write initial status
    def write_status(frame, total, state='running'):
        with open(status_path, 'w') as f:
            json.dump({
                'state': state,
                'frame': frame,
                'total_frames': total,
                'video': video_name if state == 'done' else None,
                'generation': gen,
                'log': log_file,
            }, f)

    write_status(0, n_frames)

    for step in range(total_steps):
        sim.substep()

        if step % VIEW_RENDER_EVERY == 0:
            render_frame(radius, web_palette=web_palette)
            ti.sync()

            img = sim.frame_buffer.to_numpy()
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            frames.append(img_uint8)

            frame_num = len(frames)
            # Progress output: parseable by web frontend
            print(f"FRAME {frame_num}/{n_frames}", flush=True)
            write_status(frame_num, n_frames)

    print(f"Writing video to {video_path}...")
    iio.imwrite(video_path, frames, fps=VIEW_FPS)
    write_status(n_frames, n_frames, state='done')
    print(f"DONE {video_name}")
    return video_path


def main():
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Jellyfish evolutionary optimizer")
    parser.add_argument('--gens', type=int, default=GENERATIONS,
                        help=f"Number of generations (default: {GENERATIONS})")
    parser.add_argument('--view', action='store_true',
                        help="Render best genomes as video instead of evolving")
    parser.add_argument('--gen', type=int, default=None,
                        help="Specific generation to view (use with --view)")
    parser.add_argument('--aurelia', action='store_true',
                        help="Evaluate Aurelia aurita reference genome and render video")
    parser.add_argument('--web-palette', action='store_true',
                        help="Use web frontend color palette for rendering (light bg, flat colors)")
    parser.add_argument('--sim-gen', action='store_true',
                        help="Simulate a full generation from CSV log and render video")
    parser.add_argument('--log', type=str, default='evolution_log.csv',
                        help="CSV log file to read (use with --sim-gen)")
    parser.add_argument('--frames', type=int, default=100,
                        help="Number of frames to render (use with --sim-gen, default: 100)")
    parser.add_argument('--run-id', type=str, default=None,
                        help="Run identifier for output directory (default: auto timestamp YYYYMMDD_HHMMSS)")
    parser.add_argument('--seed', type=int, default=42,
                        help="CMA-ES random seed (default: 42; use different values for independent replicate runs)")
    parser.add_argument('--instances', type=int, default=16,
                        help="Number of parallel GPU instances / CMA-ES population size (default: 16)")
    args = parser.parse_args()

    global OUTPUT_DIR
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    OUTPUT_DIR = os.path.join('output', run_id)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Run ID: {run_id}  →  {OUTPUT_DIR}/")

    if args.sim_gen:
        if args.gen is None:
            print("Error: --sim-gen requires --gen N")
            return
        sim_generation(args.gen, log_file=args.log, n_frames=args.frames,
                       web_palette=args.web_palette)
    elif args.aurelia:
        eval_aurelia(web_palette=args.web_palette)
    elif args.view:
        view_best(gen_idx=args.gen, web_palette=args.web_palette)
    else:
        evolve(args.gens, seed=args.seed)


if __name__ == "__main__":
    main()
