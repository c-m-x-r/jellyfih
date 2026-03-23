"""
tune_actuation.py — sweep actuation_strength across N instances simultaneously.

Each column in the output video is a different actuation strength, all running
the same genome (Aurelia by default, or best genome from a run).

Usage:
    uv run python tune_actuation.py
    uv run python tune_actuation.py --genome aurelia
    uv run python tune_actuation.py --genome output/20260322_231418/best_genomes.json
    uv run python tune_actuation.py --strengths 100 300 500 1000 2000 5000
"""

import os
import sys
import argparse
import numpy as np
import json

# Strengths to sweep — set via --strengths or use these defaults
DEFAULT_STRENGTHS = [50, 100, 200, 400, 800, 1600, 3200, 6400]

parser = argparse.ArgumentParser()
parser.add_argument('--strengths', type=float, nargs='+', default=DEFAULT_STRENGTHS)
parser.add_argument('--genome', type=str, default='aurelia',
                    help='aurelia | path/to/best_genomes.json')
parser.add_argument('--cycles', type=float, default=2.0,
                    help='Actuation cycles to simulate (default: 2)')
parser.add_argument('--out', type=str, default='output/tune_actuation.mp4')
args = parser.parse_args()

n_strengths = len(args.strengths)
os.environ['JELLY_INSTANCES'] = str(n_strengths)

import taichi as ti
import mpm_sim as sim
from make_jelly import fill_tank, AURELIA_GENOME

# --- Load genome ---
if args.genome == 'aurelia':
    genome = AURELIA_GENOME
    genome_label = 'Aurelia'
else:
    with open(args.genome) as f:
        history = json.load(f)
    best = max(history, key=lambda h: h['fitness'])
    genome = best['genome']
    genome_label = f"gen{best['generation']} fit={best['fitness']:.3f}"

print(f"Genome: {genome_label}")
print(f"Sweeping {n_strengths} actuation strengths: {args.strengths}")

# --- Load same genome into all instances, set different actuation strengths ---
for i in range(n_strengths):
    pos, mat, fiber, _ = fill_tank(genome, sim.n_particles)
    sim.load_particles(i, pos, mat, fiber)
    sim.instance_actuation[i] = args.strengths[i]

    # Colour by position in sweep: hue from red (weak) to violet (strong)
    sim.instance_hue[i] = 0.0 + 0.8 * (i / max(n_strengths - 1, 1))

sim.sim_time[None] = 0.0

# --- Render ---
steps_per_cycle = int(1.0 / (sim.actuation_freq * sim.dt))
total_steps = int(args.cycles * steps_per_cycle)
render_every = 500
total_frames = total_steps // render_every

print(f"Simulating {args.cycles} cycles = {total_steps} steps, {total_frames} frames")
print(f"Output: {args.out}")

import imageio.v3 as iio

os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

frames = []
for step in range(total_steps):
    sim.substep()
    sim.sim_time[None] += sim.dt

    if step % render_every == 0:
        frame_idx = step // render_every
        if frame_idx % 25 == 0:
            t = step * sim.dt
            print(f"  Frame {frame_idx}/{total_frames} (t={t:.3f}s)")

        radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)
        sim.clear_frame_buffer()
        sim.render_frame_abyss(sim.res_sub, sim.grid_side, radius)
        sim.tone_map_and_encode()
        img = sim.frame_buffer.to_numpy()
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(img_uint8)

# Add strength labels burned into first frame via numpy (simple text-free approach:
# draw a thin coloured bar at top of each column indicating relative strength)
grid_side = int(np.ceil(np.sqrt(n_strengths)))
res_sub = sim.video_res // grid_side
for frame in frames:
    for i in range(n_strengths):
        col = i % grid_side
        row = i // grid_side
        x0 = col * res_sub
        y0 = row * res_sub
        # 4px bar height proportional to log strength
        log_frac = np.log10(args.strengths[i] / min(args.strengths)) / \
                   max(np.log10(max(args.strengths) / min(args.strengths)), 1)
        bar_w = max(4, int(res_sub * log_frac))
        frame[y0:y0+6, x0:x0+bar_w] = 255  # white bar = relative strength

print(f"Writing {len(frames)} frames...")
iio.imwrite(args.out, frames, fps=30, codec='libx264', quality=8)
print(f"Done → {args.out}")
print()
print("Strengths (column order, left-to-right, top-to-bottom):")
for i, s in enumerate(args.strengths):
    print(f"  [{i}] {s:>8.0f}  (hue {sim.instance_hue[i]:.2f})")
