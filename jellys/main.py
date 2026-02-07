"""CLI entry point for the Jellys MPM simulation.

Demos:
  water     - Water settling in a box (Phase 1)
  jellyfish - Jellyfish swimming with actuation (Phase 2)
"""

import argparse
from pathlib import Path

from .config import (
    SimulationConfig,
    default_config,
    jellyfish_demo_config,
    GridConfig,
    JellyfishConfig,
    JellyfishGeometry,
    JellyfishMaterial,
    JellyfishActuation,
)
from .simulation import MPMSolver, init_taichi
from .rendering import render_simulation


def run_water_settling(config: SimulationConfig, verbose: bool = True):
    if verbose:
        print("=" * 50)
        print("Phase 1: Water Settling")
        print(f"  Grid: {config.grid.resolution}x{config.grid.resolution}")
        print(f"  Particles: {config.n_particles}")
        print(f"  Frames: {config.total_frames}")
        print("=" * 50)

    solver = MPMSolver(config)
    renderer, _ = render_simulation(
        solver, n_frames=config.total_frames,
        render_config=config.render, verbose=verbose,
    )

    if verbose:
        print(f"Output: {Path(config.render.output_dir).absolute()}")
    return renderer


def run_jellyfish_demo(config: SimulationConfig, verbose: bool = True):
    jelly = config.jellyfish
    if verbose:
        print("=" * 50)
        print("Phase 2: Jellyfish Swimming")
        print(f"  Grid: {config.grid.resolution}x{config.grid.resolution}")
        print(f"  Bell: {jelly.geometry.semi_major:.2f} x {jelly.geometry.semi_minor:.2f}")
        print(f"  Model: {jelly.material.model}, E={jelly.material.E}")
        print(f"  Actuation: {jelly.actuation.frequency} Hz, amp={jelly.actuation.amplitude}")
        print(f"  Density: {jelly.material.density}")
        print(f"  Frames: {config.total_frames}")
        print("=" * 50)

    solver = MPMSolver(config)
    if verbose:
        print(f"Water: {solver.n_water}, Jelly: {solver.n_jelly}, Total: {solver.n_particles}")

    renderer, trajectory = render_simulation(
        solver, n_frames=config.total_frames,
        render_config=config.render,
        water_level=config.physics.water_level,
        verbose=verbose, track_jellyfish=True,
    )

    if verbose and trajectory and len(trajectory) > 1:
        dy = trajectory[-1][1] - trajectory[0][1]
        print(f"\nVertical displacement: {dy:.4f}")
        if dy > 0.001:
            print("Jellyfish moved UPWARD")
        elif dy < -0.001:
            print("Jellyfish moved DOWNWARD (sinking)")
        else:
            print("Jellyfish approximately stationary")

    return renderer, trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Jellys: 2D MPM Jellyfish Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--demo", choices=["water", "jellyfish"], default="water")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--substeps", type=int, default=50)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--arch", choices=["auto", "cpu", "cuda", "vulkan"], default="auto")
    parser.add_argument("--quiet", action="store_true")

    jg = parser.add_argument_group("jellyfish")
    jg.add_argument("--jelly-model", choices=["fixed_corotated", "neo_hookean"], default="fixed_corotated")
    jg.add_argument("--jelly-stiffness", type=float, default=3000.0)
    jg.add_argument("--jelly-density", type=float, default=1.0)
    jg.add_argument("--actuation-freq", type=float, default=1.5)
    jg.add_argument("--actuation-amp", type=float, default=0.3)
    jg.add_argument("--jelly-x", type=float, default=0.5)
    jg.add_argument("--jelly-y", type=float, default=0.45)

    args = parser.parse_args()
    init_taichi(args.arch)

    if args.demo == "jellyfish":
        config = jellyfish_demo_config()
        config.jellyfish.material.model = args.jelly_model
        config.jellyfish.material.E = args.jelly_stiffness
        config.jellyfish.material.density = args.jelly_density
        config.jellyfish.material.__post_init__()
        config.jellyfish.actuation.frequency = args.actuation_freq
        config.jellyfish.actuation.amplitude = args.actuation_amp
        config.jellyfish.geometry.center = (args.jelly_x, args.jelly_y)
        if args.frames is None:
            args.frames = 210
    else:
        config = default_config()
        if args.frames is None:
            args.frames = 150

    if args.resolution != 64:
        config.grid = GridConfig(resolution=args.resolution)

    config.total_frames = args.frames
    config.physics.substeps = args.substeps
    config.render.output_dir = args.output

    if args.demo == "jellyfish":
        run_jellyfish_demo(config, verbose=not args.quiet)
    else:
        run_water_settling(config, verbose=not args.quiet)


if __name__ == "__main__":
    main()
