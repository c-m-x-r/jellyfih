#!/usr/bin/env python3
import numpy as np
import time
import cv2
from make_jelly import fill_tank, random_genome
import mpm_sim as sim

# --- EXPERIMENT SETTINGS ---
# Defined here to control the experiment
WARMUP_STEPS = 10
FRAMES = 400
SUBSTEPS = 200

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
        # Ensure grid_res matches simulation physics
        pos, mat, info = fill_tank(genome, n_particles, grid_res=int(sim.n_grid/sim.quality))
        all_positions[i] = pos
        all_materials[i] = mat
        stats.append(info)

    return all_positions, all_materials, genomes, stats

def load_population_to_gpu(all_positions, all_materials):
    print("Loading particles to GPU...")
    for i in range(all_positions.shape[0]):
        sim.load_particles(i, all_positions[i], all_materials[i])
    sim.ti.sync()
    print("GPU loading complete.")

def run_simulation_streaming(output_path="population_abyss.mp4"):
    """Run physics + Deep Sea Abyss rendering."""
    sim.sim_time[None] = 0.0
    
    print(f"Warmup: {WARMUP_STEPS} substeps...")
    for _ in range(WARMUP_STEPS):
        sim.substep()
    sim.ti.sync()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (sim.video_res, sim.video_res))

    print(f"Simulating + rendering (streaming) {FRAMES} frames...")
    start_time = time.time()

    for frame in range(FRAMES):
        for _ in range(SUBSTEPS):
            sim.substep()

        # --- ABYSS RENDER PIPELINE ---
        sim.clear_frame_buffer()
        # Pass render params: sub-resolution, grid tiles, particle radius
        sim.render_frame_abyss(sim.res_sub, sim.grid_side, 1.5)
        sim.tone_map_and_encode()
        
        frame_np = sim.frame_buffer.to_numpy()
        # Safe cast with clip to avoid 'invalid value' warnings
        frame_u8 = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))

        if frame % 10 == 0:
            print(f"  Frame {frame}/{FRAMES}")

    sim.ti.sync()
    out.release()

    elapsed = time.time() - start_time
    print(f"Sim time: {elapsed:.2f}s ({FRAMES/elapsed:.1f} fps)")
    print(f"Video saved: {output_path}")
    return elapsed

def main():
    print("=" * 50)
    print("JELLYFISH POPULATION: DEEP SEA ABYSS")
    print("=" * 50)
    
    # Generate and Load
    all_pos, all_mat, genomes, stats = generate_population(sim.n_instances, sim.n_particles)
    load_population_to_gpu(all_pos, all_mat)
    
    # Run
    run_simulation_streaming()

if __name__ == "__main__":
    main()