"""Main entry point for the Jellys MPM simulation.

Phase 1: Water settling demonstration.
Phase 2: Jellyfish swimming demonstration.
"""

import argparse
from pathlib import Path

from .config import (
    SimulationConfig,
    default_config,
    high_res_config,
    jellyfish_demo_config,
    JellyfishConfig,
    JellyfishGeometry,
    JellyfishMaterial,
    JellyfishActuation,
)
from .simulation import MPMSolver, init_taichi
from .rendering import render_simulation


def run_water_settling(config: SimulationConfig, verbose: bool = True):
    """Run the water settling simulation (Phase 1).

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
    renderer, _ = render_simulation(
        solver,
        n_frames=config.total_frames,
        render_config=config.render,
        verbose=verbose,
    )

    if verbose:
        print("\nSimulation complete!")
        print(f"Output saved to: {Path(config.render.output_dir).absolute()}")

    return renderer


def run_jellyfish_demo(config: SimulationConfig, verbose: bool = True):
    """Run the jellyfish swimming demonstration (Phase 2).

    Args:
        config: Simulation configuration with jellyfish enabled
        verbose: Print progress updates
    """
    jelly = config.jellyfish
    if verbose:
        print("=" * 50)
        print("Phase 2: The Parametric Jellyfish")
        print("=" * 50)
        print(f"Grid resolution: {config.grid.resolution}x{config.grid.resolution}")
        print(f"Water level: {config.physics.water_level:.1%}")
        print(f"Jellyfish center: ({jelly.geometry.center[0]:.2f}, {jelly.geometry.center[1]:.2f})")
        print(f"Jellyfish size: {jelly.geometry.semi_major:.2f} x {jelly.geometry.semi_minor:.2f}")
        print(f"Material model: {jelly.material.model}")
        print(f"Stiffness (E): {jelly.material.E}")
        print(f"Actuation: {jelly.actuation.frequency} Hz, amplitude {jelly.actuation.amplitude}")
        print(f"Frames: {config.total_frames}")
        print(f"Output: {config.render.output_dir}/")
        print("=" * 50)

    # Create solver
    solver = MPMSolver(config)

    if verbose:
        print(f"\nWater particles: {solver.n_water}")
        print(f"Jellyfish particles: {solver.n_jelly}")
        print(f"Total particles: {solver.n_particles}")
        print(f"\nSimulating {config.total_frames} frames...")

    # Run simulation and render with jellyfish tracking
    renderer, trajectory = render_simulation(
        solver,
        n_frames=config.total_frames,
        render_config=config.render,
        water_level=config.physics.water_level,
        verbose=verbose,
        track_jellyfish=True,
    )

    if verbose:
        print("\nSimulation complete!")
        print(f"Output saved to: {Path(config.render.output_dir).absolute()}")

        # Report jellyfish motion
        if trajectory and len(trajectory) > 1:
            start_y = trajectory[0][1]
            end_y = trajectory[-1][1]
            delta_y = end_y - start_y
            print(f"\nJellyfish vertical displacement: {delta_y:.4f}")
            if delta_y > 0:
                print("Result: Jellyfish moved UPWARD (propulsion working!)")
            elif delta_y < 0:
                print("Result: Jellyfish moved DOWNWARD (sinking)")
            else:
                print("Result: Jellyfish stayed in place (neutral)")

    return renderer, trajectory


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Jellys: 2D MPM Simulation for Soft Robotics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Demo selection
    parser.add_argument(
        "--demo",
        type=str,
        choices=["water", "jellyfish"],
        default="water",
        help="Demo to run: water settling or jellyfish swimming",
    )

    # General simulation parameters
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Grid resolution (NxN)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames (default: 150 for water, 210 for jellyfish)",
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
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # Jellyfish-specific parameters
    jelly_group = parser.add_argument_group("jellyfish options")
    jelly_group.add_argument(
        "--jelly-model",
        type=str,
        choices=["fixed_corotated", "neo_hookean"],
        default="fixed_corotated",
        help="Constitutive model for jellyfish material",
    )
    jelly_group.add_argument(
        "--jelly-stiffness",
        type=float,
        default=3000.0,
        help="Young's modulus (E) for jellyfish",
    )
    jelly_group.add_argument(
        "--jelly-density",
        type=float,
        default=1.0,
        help="Jellyfish density relative to water (1.0 = neutral)",
    )
    jelly_group.add_argument(
        "--actuation-freq",
        type=float,
        default=1.5,
        help="Contraction frequency in Hz",
    )
    jelly_group.add_argument(
        "--actuation-amp",
        type=float,
        default=0.3,
        help="Contraction amplitude (0.0-1.0)",
    )
    jelly_group.add_argument(
        "--jelly-x",
        type=float,
        default=0.5,
        help="Jellyfish spawn X position (0.0-1.0)",
    )
    jelly_group.add_argument(
        "--jelly-y",
        type=float,
        default=0.45,
        help="Jellyfish spawn Y position (0.0-1.0)",
    )

    args = parser.parse_args()

    # Initialize Taichi
    init_taichi(args.arch)

    # Build configuration based on demo type
    if args.demo == "jellyfish":
        config = jellyfish_demo_config()

        # Override jellyfish parameters from CLI
        config.jellyfish.material.model = args.jelly_model
        config.jellyfish.material.E = args.jelly_stiffness
        config.jellyfish.material.density = args.jelly_density
        config.jellyfish.material.__post_init__()  # Recompute mu, lambda

        config.jellyfish.actuation.frequency = args.actuation_freq
        config.jellyfish.actuation.amplitude = args.actuation_amp

        config.jellyfish.geometry.center = (args.jelly_x, args.jelly_y)

        # Default frames for jellyfish
        if args.frames is None:
            args.frames = 210
    else:
        config = default_config()
        if args.frames is None:
            args.frames = 150

    # Override common parameters
    if args.resolution != 64:
        config.grid.resolution = args.resolution
        config.grid.__post_init__()

    config.total_frames = args.frames
    config.physics.substeps = args.substeps
    config.render.output_dir = args.output

    # Run appropriate demo
    if args.demo == "jellyfish":
        run_jellyfish_demo(config, verbose=not args.quiet)
    else:
        run_water_settling(config, verbose=not args.quiet)


if __name__ == "__main__":
    main()
