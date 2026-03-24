"""
tall_tank.py — Render a jellyfish in a tall 128×256 tank (double height, double particles).

The domain is [0,1]×[0,2], giving the jellyfish room to swim upward without hitting
the ceiling. Grid is 128 wide × 256 tall; particle count is 160K.

Video output is 512×1024 (correct 1:2 aspect ratio, cropped from the 1024×1024 frame buffer).

Usage:
    uv run python helpers/tall_tank.py --aurelia
    uv run python helpers/tall_tank.py --gen 5
    uv run python helpers/tall_tank.py --genome "[0.05,0.04,0.18,-0.03,0.22,-0.12,0.04,0.05,0.015]"
    uv run python helpers/tall_tank.py --aurelia --cycles 6   # more swim cycles
    uv run python helpers/tall_tank.py --aurelia --palette web
"""

import os as _os
import sys as _sys

# Must be set before Taichi / mpm_sim initialise.
_os.environ.setdefault("JELLY_INSTANCES", "1")
_os.environ.setdefault("JELLY_GRID_Y",   "256")
_os.environ.setdefault("JELLY_PARTICLES","160000")
_os.environ.setdefault("JELLY_DOMAIN_H", "2.0")

import argparse
import json

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import imageio.v3 as iio

import mpm_sim as sim
from mpm_sim import WEB_PALETTE
from make_jelly import fill_tank, AURELIA_GENOME

OUTPUT_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "output")

DOMAIN_HEIGHT = 2.0
RENDER_EVERY  = 500       # substeps between frames
FPS           = 30
# Spawn slightly higher than default so jellyfish sits in the lower third of the tall tank
TALL_SPAWN    = np.array([0.5, 0.30])


def load_genome(args):
    if args.aurelia:
        print("Genome: Aurelia aurita reference")
        return AURELIA_GENOME, "aurelia"

    if args.gen is not None:
        path = _os.path.join(OUTPUT_DIR, "best_genomes.json")
        if not _os.path.exists(path):
            # Try run subdirectories
            for d in sorted(_os.listdir(OUTPUT_DIR)):
                candidate = _os.path.join(OUTPUT_DIR, d, f"best_genomes_{d}.json")
                if _os.path.exists(candidate):
                    path = candidate
                    break
        with open(path) as f:
            data = json.load(f)
        entry = next((e for e in data if e["generation"] == args.gen), None)
        if entry is None:
            entry = max(data, key=lambda e: e["generation"])
            print(f"Generation {args.gen} not found; using gen {entry['generation']}")
        genome = np.array(entry["genome"])
        print(f"Genome: gen {entry['generation']}, fitness {entry.get('fitness', '?'):.4f}")
        return genome, f"gen{entry['generation']}"

    if args.genome:
        genome = np.array(json.loads(args.genome))
        return genome, "custom"

    raise ValueError("Specify --aurelia, --gen N, or --genome [...]")


def render_frame(palette):
    if palette == "web":
        sim.clear_frame_buffer_white()
        for mat_id, r, g, b in WEB_PALETTE:
            sim.render_flat_pass(sim.video_res, 1, 3.0, mat_id, r, g, b)
    else:
        sim.clear_frame_buffer()
        sim.render_frame_abyss(sim.video_res, 1, 3.0)
        sim.tone_map_and_encode()
    raw = sim.frame_buffer.to_numpy()           # (1024, 1024, 3)
    raw = np.clip(raw, 0.0, 1.0)
    # Crop to the left 512 columns → 1024×512, then transpose to 512-wide × 1024-tall
    # frame_buffer is indexed [row, col], so [:, :512] gives rows 0-1023, cols 0-511
    cropped = raw[:, :512, :]                   # (1024, 512, 3)  — correct 1:2 aspect
    return (cropped * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Tall-tank jellyfish viewer (128×256 grid, 2-unit domain)")
    parser.add_argument("--aurelia", action="store_true", help="Use Aurelia aurita reference genome")
    parser.add_argument("--gen", type=int, default=None, help="Use best genome from generation N")
    parser.add_argument("--genome", type=str, default=None, help="JSON genome array")
    parser.add_argument("--cycles", type=int, default=3, help="Number of actuation cycles to simulate (default 3)")
    parser.add_argument("--palette", choices=["abyss", "web"], default="abyss")
    args = parser.parse_args()

    genome, label = load_genome(args)

    # Build particle set with double-height water fill
    print(f"Building particle set for tall tank (domain 1×{DOMAIN_HEIGHT})...")
    pos, mat, fiber, self_intersecting = fill_tank(
        genome,
        max_particles=sim.n_particles,
        grid_res=128,
        spawn_offset=TALL_SPAWN,
        domain_height=DOMAIN_HEIGHT,
    )
    if self_intersecting:
        print("WARNING: morphology is self-intersecting")

    n_robot  = int(np.sum(mat >= 0) - np.sum(mat == 0))
    n_water  = int(np.sum(mat == 0))
    n_muscle = int(np.sum(mat == 3))
    print(f"  {n_robot} robot particles  |  {n_water} water particles  |  {sim.n_particles} total slots")
    print(f"  {n_muscle} muscle particles")

    sim.sim_time[None] = 0.0
    sim.load_particles(0, pos, mat, fiber)

    # Simulation parameters
    steps_per_cycle = int(round(1.0 / (sim.actuation_freq * sim.dt)))
    total_steps     = steps_per_cycle * args.cycles
    print(f"Simulating {args.cycles} cycles × {steps_per_cycle} steps = {total_steps} total steps...")

    frames = []
    for step in range(total_steps):
        sim.substep()
        if step % RENDER_EVERY == 0:
            frames.append(render_frame(args.palette))
            if step % (RENDER_EVERY * 10) == 0:
                pct = 100 * step / total_steps
                print(f"  {pct:.0f}%  (step {step}/{total_steps})")

    # Final frame
    frames.append(render_frame(args.palette))

    # Write video
    out_path = _os.path.join(OUTPUT_DIR, f"tall_tank_{label}.mp4")
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    iio.imwrite(out_path, frames, fps=FPS, codec="libx264", quality=8)
    print(f"\nSaved → {out_path}  ({len(frames)} frames, {sim.video_res//2}×{sim.video_res} px)")

    # Final CoM stats
    stats = sim.get_payload_stats()
    print(f"Final payload CoM y = {stats[0,0]:.4f}  (started at ~{TALL_SPAWN[1]:.2f})")


if __name__ == "__main__":
    main()
