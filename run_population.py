#!/usr/bin/env python3
"""
Run a population of jellyfish morphologies in parallel simulation.

This script:
1. Generates N random genomes (one per simulation instance)
2. Fills tanks with water + robot particles for each
3. Loads into MPM simulation
4. Runs physics and renders video showing all morphologies
"""

import numpy as np
import time
import cv2

# Import morphology generator
from make_jelly import fill_tank, random_genome, generate_phenotype

# Import simulation (this also initializes Taichi)
import mpm_sim as sim


def generate_population(n_instances, n_particles):
    """Generate a population of filled tanks, one per instance."""
    print(f"Generating {n_instances} morphologies...")

    all_positions = np.zeros((n_instances, n_particles, 2), dtype=np.float32)
    all_materials = np.zeros((n_instances, n_particles), dtype=np.int32)
    genomes = []
    stats = []

    for i in range(n_instances):
        genome = random_genome()
        genomes.append(genome)

        pos, mat, info = fill_tank(genome, n_particles)
        all_positions[i] = pos
        all_materials[i] = mat
        stats.append(info)

        print(f"  Instance {i}: {info['n_robot']} robot, {info['n_water']} water, {info['n_dead']} dead")

    return all_positions, all_materials, genomes, stats


def load_population_to_gpu(all_positions, all_materials):
    """Load all instances into Taichi GPU fields."""
    print("Loading particles to GPU...")
    n_instances = all_positions.shape[0]

    for i in range(n_instances):
        sim.load_particles(i, all_positions[i], all_materials[i])

    sim.ti.sync()
    print("GPU loading complete.")


def run_simulation():
    """Run physics simulation and record frames."""
    print("Starting simulation...")

    # Warmup
    for _ in range(10):
        sim.substep()
    sim.ti.sync()

    start_time = time.time()

    for frame in range(sim.frames):
        for _ in range(sim.substeps_per_frame):
            sim.substep()
        sim.record_frame(frame)

        if frame % 50 == 0:
            print(f"  Frame {frame}/{sim.frames}")

    sim.ti.sync()
    elapsed = time.time() - start_time

    print(f"Simulation complete: {sim.frames} frames in {elapsed:.2f}s ({sim.frames/elapsed:.1f} fps)")
    return elapsed


def render_and_save(output_path="population_sim.mp4"):
    """Render all frames and save to video."""
    print("Rendering frames on GPU...")

    sim.clear_buffer_to_white()
    # 4x4 grid layout, 256 pixels per cell
    sim.render_all_frames(256, 4, 1.5)
    sim.ti.sync()

    print(f"Saving video to {output_path}...")

    np_video = sim.video_buffer.to_numpy()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1024, 1024))

    for f in range(sim.frames):
        frame_float = np_video[f]
        frame_u8 = (np.clip(frame_float, 0.0, 1.0) * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Video saved: {output_path}")


def visualize_genomes(genomes, output_path="population_preview.png"):
    """Save a preview image showing all morphologies."""
    import matplotlib.pyplot as plt

    n = len(genomes)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i, genome in enumerate(genomes):
        pos, mat = generate_phenotype(genome)
        ax = axes[i]

        if len(pos) > 0:
            # Water would be shown but we only have robot here
            jelly = pos[mat == 1]
            payload = pos[mat == 2]

            if len(jelly) > 0:
                ax.scatter(jelly[:, 0], jelly[:, 1], s=2, c='cyan', alpha=0.6)
            if len(payload) > 0:
                ax.scatter(payload[:, 0], payload[:, 1], s=2, c='darkred', alpha=0.8)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f"Genome {i}")
        ax.set_facecolor('lightblue')

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Preview saved: {output_path}")


def main():
    print("=" * 50)
    print("JELLYFISH POPULATION SIMULATION")
    print("=" * 50)

    # Generate population
    all_pos, all_mat, genomes, stats = generate_population(
        sim.n_instances,
        sim.n_particles
    )

    # Save preview of morphologies
    visualize_genomes(genomes)

    # Load to GPU
    load_population_to_gpu(all_pos, all_mat)

    # Run physics
    elapsed = run_simulation()

    # Render and save
    render_and_save()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("-" * 50)
    print(f"Instances:     {sim.n_instances}")
    print(f"Particles:     {sim.n_particles:,} per instance")
    print(f"Frames:        {sim.frames}")
    print(f"Sim time:      {elapsed:.2f}s")
    print(f"Performance:   {sim.frames * sim.n_instances / elapsed:.1f} instance-frames/s")
    print("=" * 50)


if __name__ == "__main__":
    main()
