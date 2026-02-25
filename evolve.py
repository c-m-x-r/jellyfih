"""
Evolutionary optimization of jellyfish morphologies using CMA-ES.

Evaluates 16 morphologies in parallel on GPU via MPM simulation.
Fitness: staying power — payload altitude maintenance against negative buoyancy.

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

import taichi as ti
import mpm_sim as sim
from make_jelly import fill_tank, random_genome, AURELIA_GENOME

# --- CONFIGURATION ---
GENERATIONS = 50
STEPS_PER_EVAL = 60000  # 3 actuation cycles at 1Hz, dt=5e-5
POPSIZE = sim.n_instances  # Must match GPU instances (16)
START_SIGMA = 0.25
OUTPUT_DIR = "output"

# Genome bounds: [cp1_x, cp1_y, cp2_x, cp2_y, end_x, end_y, t_base, t_mid, t_tip]
GENOME_LOWER = [0.0, -0.15,  0.0,  -0.2,  0.05, -0.45,  0.025,  0.025,  0.01]
GENOME_UPPER = [0.25,  0.15,  0.3,   0.15, 0.35, -0.03,  0.08,   0.1,    0.04]
GENOME_X0 = [(lo + hi) / 2 for lo, hi in zip(GENOME_LOWER, GENOME_UPPER)]

# Fitness constants
PENALTY_INVALID = 100.0

# View mode constants
VIEW_STEPS = 60000       # Sim steps for rendered view
VIEW_RENDER_EVERY = 200  # Render a frame every N substeps
VIEW_FPS = 30

# Web palette: material colors matching the web frontend (#E8F4F8, #4ECDC4, #FF6B6B, #FFA500)
# Rendered in layer order: water first (back), payload last (front)
WEB_PALETTE = [
    (0, 0.933, 0.949, 0.957),  # Water:   rgb(238, 242, 244)
    (1, 0.678, 0.882, 0.875),  # Jelly:   rgb(173, 225, 223)
    (3, 0.976, 0.757, 0.765),  # Muscle:  rgb(249, 193, 195)
    (2, 0.961, 0.824, 0.576),  # Payload: rgb(245, 210, 147)
]


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
    scores = []
    for i in range(POPSIZE):
        init_y = sim_results[i, 0]
        init_x = sim_results[i, 1]
        final_y = sim_results[i, 2]
        final_x = sim_results[i, 3]
        valid = sim_results[i, 4]
        
        if valid == 0:
            scores.append(PENALTY_INVALID)
            continue
            
        # 1. Pure Vertical Gain (Reward moving UP)
        displacement = final_y - init_y
        
        # 2. Drift Penalty (Linear is often better than quadratic for stability)
        drift = abs(final_x - init_x)
        
        # New Fitness: Move up, don't move side-to-side
        # If it sinks (negative displacement), this value drops.
        # We penalize drift heavily (e.g. 1.0m drift cancels 1.0m rise)
        fitness = displacement 
        #- (1.0 * drift)
        
        scores.append(-fitness) # Minimize negative fitness
    return scores


def load_batch(genomes):
    """
    Generate phenotypes and load all instances to GPU.
    Returns muscle_counts list and per-instance stats.
    """
    muscle_counts = []
    instance_stats = []

    for i, genome in enumerate(genomes):
        pos, mat, stats = fill_tank(
            genome, sim.n_particles, grid_res=int(sim.n_grid)
        )
        sim.load_particles(i, pos, mat)
        muscle_counts.append(stats['muscle_count'])
        instance_stats.append(stats)

    ti.sync()
    return muscle_counts, instance_stats


def run_baseline():
    """
    Run zero-actuation baseline: use a default genome, check that passive
    morphologies don't achieve significant upward displacement.
    """
    print("Running zero-actuation baseline check...")

    genome = GENOME_X0
    genomes = [genome] * POPSIZE
    muscle_counts, _ = load_batch(genomes)

    # Run a shorter sim for baseline (1 cycle)
    results = sim.run_batch_headless(20000)

    displacement = results[0, 2] - results[0, 0]
    print(f"  Baseline displacement: {displacement:.4f}")
    if abs(displacement) > 0.05:
        print(f"  WARNING: Passive displacement is {displacement:.4f}. "
              f"This may indicate buoyancy/gravity artifacts dominating fitness.")
    else:
        print(f"  OK: Passive displacement is small ({displacement:.4f})")

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
        # Single generation: fill all 4 rows with the same genome
        entry = None
        for h in history:
            if h['generation'] == gen_idx:
                entry = h
                break
        if entry is None:
            print(f"Generation {gen_idx} not found in history.")
            return
        row_entries = [entry] * grid_side
        label = f"gen_{gen_idx}"
        print(f"Viewing best genome from generation {gen_idx} "
              f"(fitness: {entry['fitness']:.4f})")
    else:
        # Pick up to 4 evenly spaced generations
        n_avail = len(history)
        if n_avail >= grid_side:
            indices = np.linspace(0, n_avail - 1, grid_side, dtype=int)
        else:
            indices = list(range(n_avail))
        row_entries = [history[i] for i in indices]
        # Pad with last entry if fewer than 4 generations
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


def evolve(generations):
    """Run the evolutionary loop."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint.pkl")
    csv_path = os.path.join(OUTPUT_DIR, "evolution_log.csv")
    json_path = os.path.join(OUTPUT_DIR, "best_genomes.json")

    # Try to resume from checkpoint
    resumed = load_checkpoint(checkpoint_path)
    if resumed:
        es, start_gen, history = resumed
        start_gen += 1  # Continue from next generation
    else:
        # Run baseline check first
        run_baseline()

        # Initialize CMA-ES with built-in bounds handling
        # CMA_stds normalizes the search space so mutations are proportional
        # to each gene's range (e.g. t_tip range 0.03 vs end_y range 0.42)
        gene_ranges = [hi - lo for lo, hi in zip(GENOME_LOWER, GENOME_UPPER)]
        opts = {
            'popsize': POPSIZE,
            'bounds': [GENOME_LOWER, GENOME_UPPER],
            'seed': 42,
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
                 ['fitness', 'final_y', 'displacement', 'drift', 'muscle_count', 'valid', 'sigma']
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
        n_invalid = sum(1 for f in fitness_values if f >= PENALTY_INVALID)
        n_invalid_total += n_invalid

        # Per-individual CSV logging
        for ind in range(POPSIZE):
            disp = sim_results[ind, 2] - sim_results[ind, 0]
            drift = abs(sim_results[ind, 3] - sim_results[ind, 1])
            valid = sim_results[ind, 4]
            final_y = sim_results[ind, 2]
            row = [gen, ind] + list(genomes[ind]) + [
                -fitness_values[ind], final_y, disp, drift,
                muscle_counts[ind], int(valid), es.sigma
            ]
            csv_writer.writerow(row)
        csv_file.flush()

        # Best genome per generation
        history.append({
            'generation': gen,
            'genome': genomes[best_idx].tolist(),
            'fitness': float(best_fitness),
            'displacement': float(best_disp),
            'sigma': float(es.sigma),
        })
        with open(json_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Console output
        best_alt = sim_results[best_idx, 2]
        si_str = f" SI: {n_self_intersect}" if n_self_intersect > 0 else ""
        print(f"Gen {gen:3d} | Best: {best_fitness:+.4f} | "
              f"Alt: {best_alt:.4f} | "
              f"Disp: {best_disp:+.4f} | "
              f"Sigma: {es.sigma:.4f} | "
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

    csv_path = os.path.join(OUTPUT_DIR, log_file)
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

    # Rendering setup
    log_stem = os.path.splitext(log_file)[0]
    video_name = f"sim_{log_stem}_gen{gen}.mp4"
    video_path = os.path.join(OUTPUT_DIR, video_name)
    status_path = os.path.join(OUTPUT_DIR, "sim_status.json")

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
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        evolve(args.gens)


if __name__ == "__main__":
    main()
