"""Main entry point for the Jellys MPM simulation.

Phase 1: Water settling demonstration.
"""

import argparse
from pathlib import Path

from .config import SimulationConfig, default_config, high_res_config
from .simulation import MPMSolver, init_taichi
from .rendering import render_simulation


def run_water_settling(config: SimulationConfig, verbose: bool = True):
    """Run the water settling simulation.

    Args:
        config: Simulation configuration
        verbose: Print progress updates
    """
    if verbose:
        print("=" * 50)
        print("Phase 1: The Differentiable Aquarium")
        print("=" * 50)
        print(f"Grid resolution: {config.grid.resolution}x{config.grid.resolution}")
        print(f"Particles: {config.n_particles}")
        print(f"Frames: {config.total_frames}")
        print(f"Substeps per frame: {config.steps_per_frame}")
        print(f"Output: {config.render.output_dir}/")
        print("=" * 50)

    # Create solver
    solver = MPMSolver(config)

    if verbose:
        print(f"\nSimulating {config.total_frames} frames...")

    # Run simulation and render
    renderer = render_simulation(
        solver,
        n_frames=config.total_frames,
        render_config=config.render,
        verbose=verbose,
    )

    if verbose:
        print("\nSimulation complete!")
        print(f"Output saved to: {Path(config.render.output_dir).absolute()}")

    return renderer


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Jellys: 2D MPM Fluid Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Grid resolution (NxN)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=150,
        help="Number of frames to simulate",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=50,
        help="Substeps per frame",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for frames and GIF",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["auto", "cpu", "cuda", "vulkan"],
        default="auto",
        help="Taichi backend architecture",
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high resolution preset (128x128 grid)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Initialize Taichi
    init_taichi(args.arch)

    # Build configuration
    if args.high_res:
        config = high_res_config()
    else:
        config = default_config()

    # Override with CLI arguments
    if args.resolution != 64:
        config.grid.resolution = args.resolution
        config.grid.__post_init__()  # Recompute dx, inv_dx

    config.total_frames = args.frames
    config.physics.substeps = args.substeps
    config.render.output_dir = args.output

    # Run simulation
    run_water_settling(config, verbose=not args.quiet)


if __name__ == "__main__":
    main()
