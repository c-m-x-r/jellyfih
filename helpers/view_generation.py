"""
view_generation.py — Render all individuals from one generation side-by-side.

Each of the 16 GPU instances shows a different individual from the chosen
generation, sorted by fitness (best first). Useful for inspecting the full
diversity of a generation rather than just the elite.

Usage:
    uv run python helpers/view_generation.py --gen 5
    uv run python helpers/view_generation.py --gen 5 --include-invalid
    uv run python helpers/view_generation.py --gen 5 --sort index
    uv run python helpers/view_generation.py --gen 5 --palette web
    uv run python helpers/view_generation.py --gen 5 --log output/myrun/evolution_log_myrun.csv
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti
import imageio.v3 as iio

import mpm_sim as sim
from mpm_sim import WEB_PALETTE
from make_jelly import fill_tank

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
RENDER_EVERY = 500
FPS = 30
COL_HUES = [0.22, 0.33, 0.44, 0.50]


def random_contrasting_hues():
    """Return (jelly_hue, muscle_hue) as a roughly complementary pair."""
    h_jelly = random.random()
    h_muscle = (h_jelly + 0.45 + random.uniform(-0.05, 0.05)) % 1.0
    return h_jelly, h_muscle


def load_generation_individuals(csv_path, gen, sort_by="fitness", include_invalid=False):
    """
    Read all individuals for the given generation from an evolution log CSV.

    Returns a list of dicts:
        {'individual': int, 'genome': list[float], 'fitness': float, 'valid': bool}
    Sorted by sort_by ('fitness' desc or 'index' asc).
    """
    individuals = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("generation") in ("generation", None):
                continue  # skip duplicate headers
            try:
                if int(row["generation"]) != gen:
                    continue
            except ValueError:
                continue

            valid = row.get("valid", "1") not in ("0", "False", "false", "")
            if not include_invalid and not valid:
                continue

            genome = [float(row[f"gene_{i}"]) for i in range(9)]
            individuals.append({
                "individual": int(row["individual"]),
                "genome": genome,
                "fitness": float(row["fitness"]),
                "valid": valid,
            })

    if sort_by == "fitness":
        individuals.sort(key=lambda r: r["fitness"], reverse=True)
    else:
        individuals.sort(key=lambda r: r["individual"])

    return individuals


def find_csv(log_arg):
    """Resolve the evolution log CSV path from the --log argument or defaults."""
    if log_arg:
        if not os.path.exists(log_arg):
            print(f"ERROR: log file not found: {log_arg}")
            sys.exit(1)
        return log_arg

    default = os.path.join(OUTPUT_DIR, "evolution_log.csv")
    if os.path.exists(default):
        return default

    # Search for any evolution_log CSV in output subdirectories
    candidates = []
    for root, _dirs, files in os.walk(OUTPUT_DIR):
        for fname in files:
            if fname.startswith("evolution_log") and fname.endswith(".csv"):
                candidates.append(os.path.join(root, fname))
    if candidates:
        # Most recently modified
        candidates.sort(key=os.path.getmtime, reverse=True)
        print(f"Using log: {candidates[0]}")
        return candidates[0]

    print(f"ERROR: no evolution_log CSV found in {OUTPUT_DIR}/. Run evolve.py first.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Render all individuals from one generation in a 4x4 grid."
    )
    parser.add_argument("--gen", type=int, required=True,
                        help="Generation number to view")
    parser.add_argument("--sort", choices=["fitness", "index"], default="fitness",
                        help="Sort order within grid (default: fitness desc = best first)")
    parser.add_argument("--include-invalid", action="store_true",
                        help="Include invalid individuals (default: skip them)")
    parser.add_argument("--palette", choices=["abyss", "web", "random"], default="abyss",
                        help="Colour scheme: abyss (dark HDR), web (flat light), "
                             "random (each instance gets its own contrasting pair). Default: abyss")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to evolution_log CSV (default: output/evolution_log.csv)")
    parser.add_argument("--steps", type=int, default=60000,
                        help="Simulation substeps (default: 60000 = 3 cycles)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 path (default: output/view_gen_N.mp4)")
    args = parser.parse_args()

    csv_path = find_csv(args.log)
    individuals = load_generation_individuals(
        csv_path, args.gen,
        sort_by=args.sort,
        include_invalid=args.include_invalid,
    )

    if not individuals:
        label = "including invalid" if args.include_invalid else "valid only"
        print(f"No individuals found for generation {args.gen} ({label}) in {csv_path}")
        sys.exit(1)

    n_total = len(individuals)
    n_shown = min(n_total, sim.n_instances)
    showing_str = (f"showing {n_shown}/{n_total}"
                   if n_total > sim.n_instances else f"all {n_total}")
    print(f"Generation {args.gen}: {n_total} individuals loaded — {showing_str} "
          f"(sorted by {args.sort}{'  [all validity]' if args.include_invalid else ''})")
    if n_total > sim.n_instances:
        print(f"  NOTE: generation has {n_total} individuals but GPU batch is "
              f"{sim.n_instances}; displaying top {n_shown} by {args.sort}.")
    f_vals = [r["fitness"] for r in individuals[:n_shown]]
    print(f"  Fitness of shown: best={f_vals[0]:.4f}  worst={f_vals[-1]:.4f}  "
          f"mean={np.mean(f_vals):.4f}")

    # Pad to 16 GPU slots with the last entry
    genomes = [np.array(ind["genome"]) for ind in individuals]
    while len(genomes) < sim.n_instances:
        genomes.append(genomes[-1])
    genomes = genomes[:sim.n_instances]

    # Load each genome as a separate GPU instance
    print(f"Generating {sim.n_instances} phenotypes...", flush=True)
    for i, genome in enumerate(genomes):
        pos, mat, fiber, stats = fill_tank(genome, sim.n_particles)
        sim.load_particles(i, pos, mat, fiber)
    ti.sync()

    # Configure per-instance colours
    if args.palette == "random":
        # Each instance gets its own contrasting pair — reveals morphology variety at a glance
        for m in range(sim.n_instances):
            h_jelly, h_muscle = random_contrasting_hues()
            sim.instance_hue[m] = h_jelly
            sim.instance_muscle_hue[m] = h_muscle
        print("  Palette: random (each cell has its own colour pair)")
    elif args.palette == "abyss":
        for m in range(sim.n_instances):
            sim.instance_hue[m] = COL_HUES[m % sim.grid_side]
            sim.instance_muscle_hue[m] = (COL_HUES[m % sim.grid_side] + 0.15) % 1.0

    output_path = args.output or os.path.join(OUTPUT_DIR, f"view_gen_{args.gen}.mp4")
    total_frames = args.steps // RENDER_EVERY
    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)

    print(f"Rendering {total_frames} frames "
          f"({args.steps} steps, every {RENDER_EVERY} substeps, palette: {args.palette})...")

    sim.sim_time[None] = 0.0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = []

    for step in range(args.steps):
        sim.substep()

        if step % RENDER_EVERY == 0:
            if args.palette == "web":
                sim.clear_frame_buffer_white()
                for mat_id, r, g, b in WEB_PALETTE:
                    sim.render_flat_pass(sim.res_sub, sim.grid_side, radius, mat_id, r, g, b)
            else:  # abyss or random (both use HDR abyss pipeline)
                sim.clear_frame_buffer()
                sim.render_frame_abyss(sim.res_sub, sim.grid_side, radius)
                sim.tone_map_and_encode()
            ti.sync()

            img = sim.frame_buffer.to_numpy()
            frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))

    iio.imwrite(output_path, frames, fps=FPS)
    print(f"Done — {len(frames)} frames written to {output_path}")


if __name__ == "__main__":
    main()
