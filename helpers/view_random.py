"""
view_random.py — Render 16 independently random jellyfish in a 4x4 grid.

Each GPU instance gets a distinct random genome and a random colour pair.

Usage:
    uv run python helpers/view_random.py
    uv run python helpers/view_random.py --palette abyss
    uv run python helpers/view_random.py --flow
    uv run python helpers/view_random.py --flow --palette abyss
    uv run python helpers/view_random.py --steps 60000
    uv run python helpers/view_random.py --seed 42
"""

import os
import sys
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti
import imageio.v3 as iio

import mpm_sim as sim
from mpm_sim import WEB_PALETTE
from make_jelly import fill_tank, random_genome

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
RENDER_EVERY = 500
FPS = 30


def random_contrasting_hues():
    h_jelly = random.random()
    h_muscle = (h_jelly + 0.45 + random.uniform(-0.05, 0.05)) % 1.0
    return h_jelly, h_muscle


def main():
    parser = argparse.ArgumentParser(
        description="Render 16 random jellyfish in a 4x4 grid."
    )
    parser.add_argument("--palette", choices=["abyss", "web", "random"], default="random",
                        help="Colour scheme (default: random)")
    parser.add_argument("--flow", action="store_true",
                        help="Add vorticity overlay (abyss/random palettes only)")
    parser.add_argument("--steps", type=int, default=60000,
                        help="Simulation substeps (default: 60000 = 3 cycles)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 path (default: output/view_random.mp4)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    n = sim.n_instances
    genomes = [random_genome() for _ in range(n)]
    print(f"Generated {n} random genomes")
    for i, g in enumerate(genomes):
        print(f"  [{i:2d}] {np.round(g, 3)}")

    print(f"\nGenerating {n} phenotypes...", flush=True)
    for i, genome in enumerate(genomes):
        pos, mat, fiber, stats = fill_tank(genome, sim.n_particles)
        sim.load_particles(i, pos, mat, fiber)
        print(f"  [{i:2d}] muscle={stats['muscle_count']}  "
              f"jelly={stats['n_jelly']}  payload={int(np.sum(mat == 2))}")
    ti.sync()

    # Per-instance colours
    if args.palette == "random":
        for m in range(n):
            h_jelly, h_muscle = random_contrasting_hues()
            sim.instance_hue[m] = h_jelly
            sim.instance_muscle_hue[m] = h_muscle
        print("  Palette: random (each cell has its own colour pair)")
    elif args.palette == "abyss":
        col_hues = [0.22, 0.33, 0.44, 0.50]
        for m in range(n):
            sim.instance_hue[m] = col_hues[m % sim.grid_side]
            sim.instance_muscle_hue[m] = (col_hues[m % sim.grid_side] + 0.15) % 1.0

    use_flow = args.flow and args.palette != "web"
    if args.flow and args.palette == "web":
        print("  NOTE: --flow ignored with --palette web")

    output_path = args.output or os.path.join(OUTPUT_DIR, "view_random.mp4")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_frames = args.steps // RENDER_EVERY
    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)

    palette_label = args.palette + ("+flow" if use_flow else "")
    print(f"\nRendering {total_frames} frames "
          f"({args.steps} steps, every {RENDER_EVERY} substeps, palette: {palette_label})...")

    sim.sim_time[None] = 0.0
    frames = []

    for step in range(args.steps):
        sim.substep()

        if step % RENDER_EVERY == 0:
            if args.palette == "web":
                sim.clear_frame_buffer_white()
                for mat_id, r, g, b in WEB_PALETTE:
                    sim.render_flat_pass(sim.res_sub, sim.grid_side, radius, mat_id, r, g, b)
            else:
                sim.clear_frame_buffer()
                sim.render_frame_abyss(sim.res_sub, sim.grid_side, radius)
                if use_flow:
                    sim.render_vorticity_overlay(sim.res_sub, sim.grid_side, 0.002)
                sim.tone_map_and_encode()
            ti.sync()

            img = sim.frame_buffer.to_numpy()
            frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))

    iio.imwrite(output_path, frames, fps=FPS)
    print(f"Done — {len(frames)} frames written to {output_path}")


if __name__ == "__main__":
    main()
