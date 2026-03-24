"""
view_single.py — Render a single genome at full resolution.

By default runs with a single GPU instance so the whole 1024x1024 frame
shows one jellyfish in full detail. Use --instances 16 to get the classic
4x4 colour-variant grid.

Wall time: ~4-5 s for 60 K steps regardless of instance count (GPU bound).

Usage:
    uv run python helpers/view_single.py --aurelia
    uv run python helpers/view_single.py --gen 5
    uv run python helpers/view_single.py --genome "[0.05,0.04,0.18,-0.03,0.22,-0.12,0.04,0.05,0.015]"
    uv run python helpers/view_single.py --aurelia --flow            # vorticity overlay
    uv run python helpers/view_single.py --aurelia --palette web
    uv run python helpers/view_single.py --aurelia --palette random  # two contrasting colours
    uv run python helpers/view_single.py --aurelia --no-payload      # payloadless symmetry
    uv run python helpers/view_single.py --gen 3 --instances 16      # 4x4 colour grid
"""

# JELLY_INSTANCES must be set before Taichi initialises (at mpm_sim import time).
# We do a minimal early parse of --instances here, before any other imports.
import os as _os
import sys as _sys
import argparse as _ap

_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument("--instances", type=int, default=1)
_pre_args, _ = _pre.parse_known_args()
_os.environ.setdefault("JELLY_INSTANCES", str(_pre_args.instances))

# Full imports (Taichi / mpm_sim initialise here, picking up JELLY_INSTANCES)
import argparse
import json
import random

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import taichi as ti
import imageio.v3 as iio

import mpm_sim as sim
from mpm_sim import WEB_PALETTE
from make_jelly import fill_tank, AURELIA_GENOME, random_genome

OUTPUT_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "output")
RENDER_EVERY = 500
FPS = 30
COL_HUES = [0.22, 0.33, 0.44, 0.50]


def random_contrasting_hues():
    """Return (jelly_hue, muscle_hue) as a roughly complementary pair."""
    h_jelly = random.random()
    h_muscle = (h_jelly + 0.45 + random.uniform(-0.05, 0.05)) % 1.0
    return h_jelly, h_muscle


def load_genome(args):
    """Resolve genome from CLI flags. Returns (genome_array, label_string)."""
    if args.aurelia:
        print("Genome: Aurelia aurita reference")
        return AURELIA_GENOME, "aurelia"

    if args.gen is not None:
        # Search default path and any run subdirectories
        path = _os.path.join(OUTPUT_DIR, "best_genomes.json")
        if not _os.path.exists(path):
            for root, _, files in _os.walk(OUTPUT_DIR):
                for fname in files:
                    if fname.startswith("best_genomes") and fname.endswith(".json"):
                        path = _os.path.join(root, fname)
                        break
        if not _os.path.exists(path):
            print("ERROR: best_genomes JSON not found. Run evolve.py first.")
            _sys.exit(1)
        with open(path) as f:
            records = json.load(f)
        matches = [r for r in records if r["generation"] == args.gen]
        if not matches:
            print(f"ERROR: No record for generation {args.gen} in {path}.")
            _sys.exit(1)
        rec = matches[0]
        print(f"Genome: generation {args.gen} best  (fitness={rec['fitness']:.4f})")
        return np.array(rec["genome"]), f"gen{args.gen}"

    if args.genome:
        try:
            g = np.array(json.loads(args.genome), dtype=float)
        except Exception as e:
            print(f"ERROR parsing --genome: {e}")
            _sys.exit(1)
        if len(g) != 9:
            print(f"ERROR: genome must have 9 values, got {len(g)}")
            _sys.exit(1)
        print(f"Genome: from CLI  {g.round(4)}")
        return g, "custom"

    genome = random_genome()
    print(f"Genome: random  {genome.round(4)}")
    return genome, "random"


def main():
    parser = argparse.ArgumentParser(
        description="Render a single genome. Default: 1 instance = full 1024x1024 view."
    )
    parser.add_argument("--aurelia", action="store_true",
                        help="Use the Aurelia aurita reference genome")
    parser.add_argument("--gen", type=int, default=None,
                        help="Load best genome from generation N")
    parser.add_argument("--genome", type=str, default=None,
                        help='JSON genome array, e.g. "[0.05, 0.04, ...]"')
    parser.add_argument("--no-payload", action="store_true",
                        help="Omit the payload block (payloadless symmetry test)")
    parser.add_argument("--palette", choices=["abyss", "web", "random"], default="abyss",
                        help="Colour scheme: abyss (dark HDR), web (flat light), "
                             "random (two contrasting colours). Default: abyss")
    parser.add_argument("--flow", action="store_true",
                        help="Add vorticity overlay (abyss/random palettes only)")
    parser.add_argument("--steps", type=int, default=60000,
                        help="Simulation substeps (default: 60000 = 3 cycles)")
    parser.add_argument("--instances", type=int, default=_pre_args.instances,
                        help="GPU instances (default: 1 = single full-res; "
                             "16 = 4x4 colour-variant grid)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 path (default: output/view_single_<label>.mp4)")
    args = parser.parse_args()

    genome, label = load_genome(args)
    with_payload = not args.no_payload
    suffix = "" if with_payload else "_nopayload"
    output_path = args.output or _os.path.join(OUTPUT_DIR, f"view_single_{label}{suffix}.mp4")

    # Generate phenotype once; broadcast to all GPU instances
    print(f"Generating phenotype (with_payload={with_payload})...", flush=True)
    pos, mat, fiber, stats = fill_tank(genome, sim.n_particles, with_payload=with_payload)
    print(f"  jelly={stats['n_jelly']}  muscle={stats['muscle_count']}  "
          f"payload={int(np.sum(mat == 2))}  water={stats['n_water']}")

    for i in range(sim.n_instances):
        sim.load_particles(i, pos, mat, fiber)
    ti.sync()

    # Configure per-instance colours
    if args.palette == "random":
        h_jelly, h_muscle = random_contrasting_hues()
        print(f"  Palette: random — jelly={h_jelly:.3f}, muscle={h_muscle:.3f}")
        for m in range(sim.n_instances):
            sim.instance_hue[m] = h_jelly
            sim.instance_muscle_hue[m] = h_muscle
    elif args.palette == "abyss":
        for m in range(sim.n_instances):
            sim.instance_hue[m] = COL_HUES[m % sim.grid_side]
            sim.instance_muscle_hue[m] = (COL_HUES[m % sim.grid_side] + 0.15) % 1.0
    # web palette: colours defined per-material in WEB_PALETTE, no per-instance hues needed

    sim.sim_time[None] = 0.0
    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)
    total_frames = args.steps // RENDER_EVERY
    use_flow = args.flow and args.palette != "web"

    palette_label = args.palette + ("+flow" if use_flow else "")
    print(f"Rendering {total_frames} frames "
          f"({args.steps} steps, every {RENDER_EVERY} substeps, "
          f"palette: {palette_label}, instances: {sim.n_instances})...")

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = []

    for step in range(args.steps):
        sim.substep()

        if step % RENDER_EVERY == 0:
            if args.palette == "web":
                sim.clear_frame_buffer_white()
                for mat_id, r, g, b in WEB_PALETTE:
                    sim.render_flat_pass(sim.res_sub, sim.grid_side, radius, mat_id, r, g, b)
            else:  # abyss or random (both use the abyss HDR pipeline)
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
