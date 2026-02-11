import numpy as np
import cma
from make_jelly import fill_tank, random_genome
import mpm_sim as sim

def objective_function(genomes):
    # 1. Map Genomes to Particles (CPU)
    # This happens on CPU, but is fast enough for 16 instances
    all_pos = []
    all_mat = []
    
    for g in genomes:
        # Note: Enforce bounds here if CMA-ES doesn't!
        pos, mat, _ = fill_tank(g, sim.n_particles, grid_res=128)
        all_pos.append(pos)
        all_mat.append(mat)

    # 2. Upload to GPU (Batch Load)
    sim.reset_state() # Clear previous run
    for i in range(sim.n_instances):
        sim.load_particles(i, all_pos[i], all_mat[i])
    
    # 3. Run Simulation (Headless)
    # e.g., run for 2000 substeps
    sim.run_batch_headless(steps=2000) 
    
    # 4. Compute Fitness
    # Fetch final data needed for calculation
    # (Assuming you calculate fitness based on payload displacement)
    final_x = sim.x.to_numpy() 
    final_mat = sim.material.to_numpy()
    
    scores = calculate_fitness_batch(final_x, final_mat)
    
    return scores # minimize cost of transport

def main():
    # Initialize CMA-ES
    # 9 genes, initial sigma 0.1
    es = cma.CMAEvolutionStrategy(9 * [0.5], 0.1, {'popsize': sim.n_instances})

    while not es.stop():
        solutions = es.ask() # Get list of new genomes
        fitnesses = objective_function(solutions)
        es.tell(solutions, fitnesses)
        es.disp()
        
        # Optional: Render the best guy every 10 gens
        if es.countiter % 10 == 0:
            save_checkpoint_video(es.best.x)

if __name__ == "__main__":
    main()