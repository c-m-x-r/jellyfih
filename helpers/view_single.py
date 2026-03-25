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

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        if len(g) not in (9, 11):
            print(f"ERROR: genome must have 9 or 11 values, got {len(g)}")
            _sys.exit(1)
        print(f"Genome: from CLI  {g.round(4)}")
        return g, "custom"

    genome = random_genome()
    print(f"Genome: random  {genome.round(4)}")
    return genome, "random"


def _save_payload_track(times, ys, label, video_path):
    """Write payload CoM trajectory to CSV and a matplotlib plot."""
    base = video_path.replace(".mp4", "")
    csv_path  = base + "_payload_track.csv"
    plot_path = base + "_payload_track.png"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "payload_com_y"])
        w.writerows(zip(times, ys))
    print(f"  Track CSV → {csv_path}")

    t = np.array(times)
    y = np.array(ys)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Cycle boundary lines
    cycle_period = 1.0 / sim.actuation_freq
    t_max = t[-1]
    cycle_n = int(t_max / cycle_period) + 1
    for c in range(1, cycle_n):
        ax.axvline(c * cycle_period, color="0.75", linewidth=0.8, linestyle="--", zorder=1)

    ax.plot(t, y, color="#e87d52", linewidth=1.6, zorder=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Payload CoM  y")
    ax.set_title(f"Payload vertical position — {label}")
    ax.set_xlim(t[0], t[-1])

    # Annotate net rise
    net = y[-1] - y[0]
    sign = "+" if net >= 0 else ""
    ax.annotate(f"net {sign}{net:.4f}", xy=(t[-1], y[-1]),
                xytext=(-8, 8), textcoords="offset points",
                ha="right", fontsize=9, color="#444")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Track plot → {plot_path}")


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
                        help="Also render a white-bg RdBu vorticity video alongside the main output")
    parser.add_argument("--rainbow", action="store_true",
                        help="Colour water particles by velocity direction (full hue wheel)")
    parser.add_argument("--steps", type=int, default=60000,
                        help="Simulation substeps (default: 60000 = 3 cycles)")
    parser.add_argument("--instances", type=int, default=_pre_args.instances,
                        help="GPU instances (default: 1 = single full-res; "
                             "16 = 4x4 colour-variant grid)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 path (default: output/view_single_<label>.mp4)")
    parser.add_argument("--track", action="store_true",
                        help="Record payload CoM y over time → CSV + plot")
    args = parser.parse_args()

    genome, label = load_genome(args)
    with_payload = not args.no_payload
    suffix = "" if with_payload else "_nopayload"
    output_path = args.output or _os.path.join(OUTPUT_DIR, f"view_single_{label}{suffix}.mp4")

    # Generate phenotype once; broadcast to all GPU instances
    axisym_mode = _os.environ.get('JELLY_AXISYM', '0') == '1'
    print(f"Generating phenotype (with_payload={with_payload}"
          + (", AXISYM" if axisym_mode else "") + ")...", flush=True)
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
    use_flow = args.flow and args.palette != "web"
    sim.water_angle_color[None] = 2 if args.rainbow else 1   # 2=rainbow, 1=plain blue

    radius = max(1.0, sim.res_sub / sim.n_grid * 0.6)
    total_frames = args.steps // RENDER_EVERY
    vort_path = output_path.replace(".mp4", "_vorticity.mp4")

    palette_label = args.palette + ("+flow" if use_flow else "")
    print(f"Rendering {total_frames} frames "
          f"({args.steps} steps, every {RENDER_EVERY} substeps, "
          f"palette: {palette_label}, instances: {sim.n_instances})...")
    if use_flow:
        print(f"  Vorticity video → {vort_path}")

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = []
    frames_vort = []
    track_t = []
    track_y = []

    for step in range(args.steps):
        sim.substep()

        if step % RENDER_EVERY == 0:
            if args.track:
                payload_stats = sim.get_payload_stats()
                track_t.append(sim.sim_time[None])
                track_y.append(float(payload_stats[0, 0]))

            # --- main video ---
            if args.palette == "web":
                sim.clear_frame_buffer_white()
                for mat_id, r, g, b in WEB_PALETTE:
                    sim.render_flat_pass(sim.res_sub, sim.grid_side, radius, mat_id, r, g, b)
            else:
                sim.clear_frame_buffer()
                sim.render_frame_abyss(sim.res_sub, sim.grid_side, radius)
                sim.tone_map_and_encode()
            ti.sync()
            frames.append((np.clip(sim.frame_buffer.to_numpy(), 0, 1) * 255).astype(np.uint8))

            # --- white-bg vorticity video (only when --flow) ---
            if use_flow:
                sim.compute_vorticity_grid(0.10)
                sim.render_vorticity_white(sim.res_sub, sim.grid_side, 0.85)
                ti.sync()
                frames_vort.append((np.clip(sim.frame_buffer.to_numpy(), 0, 1) * 255).astype(np.uint8))

    iio.imwrite(output_path, frames, fps=FPS)
    print(f"Done — {len(frames)} frames written to {output_path}")

    if use_flow:
        iio.imwrite(vort_path, frames_vort, fps=FPS)
        print(f"      {len(frames_vort)} vorticity frames written to {vort_path}")

    if args.track and track_t:
        _save_payload_track(track_t, track_y, label, output_path)


if __name__ == "__main__":
    main()
