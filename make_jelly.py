import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.spatial import cKDTree

# --- CONFIGURATION ---
PAYLOAD_WIDTH = 0.15   # Width of the central rigid payload
PAYLOAD_HEIGHT = 0.1   # Height of the payload
DENSITY_RES = 128      # Grid resolution for rasterizing particles (higher = more particles)
DEFAULT_SPAWN = np.array([0.5, 0.7])  # Default spawn position in normalized coords

def cubic_bezier(p0, p1, p2, p3, t):
    """Returns a point on the cubic Bezier curve at time t."""
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

def get_normals_2d(points):
    """Calculates normal vectors for a 2D line strip."""
    # Tangents
    tangents = np.gradient(points, axis=0)
    # Normals are perpendicular to tangents (-y, x)
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    # Normalize
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    # Avoid division by zero
    norm[norm == 0] = 1
    return normals / norm

def generate_phenotype(genome, spawn_offset=None):
    """
    Converts a CMA-ES genome vector into particle positions and materials.

    Args:
        genome: 9-element vector defining bell shape
        spawn_offset: [x, y] position for payload center (default: [0.5, 0.7])

    Genome Vector Format (9 genes):
    [0] cp1_x: Control Point 1 X (relative to start)
    [1] cp1_y: Control Point 1 Y (relative to start)
    [2] cp2_x: Control Point 2 X
    [3] cp2_y: Control Point 2 Y
    [4] end_x: Tip X
    [5] end_y: Tip Y
    [6] t_base: Thickness at connection point
    [7] t_mid:  Thickness at middle of bell
    [8] t_tip:  Thickness at tip of bell

    Returns:
        positions: (N, 2) array of particle positions
        materials: (N,) array of material IDs (1=jelly, 2=payload)
    """
    if spawn_offset is None:
        spawn_offset = DEFAULT_SPAWN
    
    # 1. Define Key Points
    # Start point is fixed at the corner of the payload
    start_p = np.array([PAYLOAD_WIDTH / 2.0, 0.0]) 
    
    # Unpack genome (Constraint: Keep X positive to stay on right side)
    cp1 = start_p + np.array([abs(genome[0]), genome[1]])
    cp2 = start_p + np.array([abs(genome[2]), genome[3]])
    end_p = start_p + np.array([abs(genome[4]), genome[5]])
    
    # Thickness inputs (ensure positive)
    t_base = abs(genome[6])
    t_mid = abs(genome[7])
    t_tip = abs(genome[8])
    
    # 2. Generate Spine Curve
    t_steps = np.linspace(0, 1, 50)
    spine_points = np.array([cubic_bezier(start_p, cp1, cp2, end_p, t) for t in t_steps])
    
    # 3. Generate Envelope (Flesh)
    normals = get_normals_2d(spine_points)
    
    # Interpolate thickness along the curve
    # Quadratic thickness profile: Base -> Mid -> Tip
    t_profile = np.interp(t_steps, [0, 0.5, 1], [t_base, t_mid, t_tip])
    
    # Create outer and inner edges
    # Note: We subtract/add normal * thickness/2 to center flesh on spine
    outer_curve = spine_points + normals * (t_profile[:, None] / 2.0)
    inner_curve = spine_points - normals * (t_profile[:, None] / 2.0)
    
    # Combine into a closed polygon (Right Side)
    # Trace outer forward, then inner backward to close loop
    polygon_points = np.vstack([outer_curve, inner_curve[::-1]])
    
    # 4. Rasterize (Fill Polygon with Particles)
    # Create a bounding box scan
    min_x, min_y = np.min(polygon_points, axis=0)
    max_x, max_y = np.max(polygon_points, axis=0)
    
    x_range = np.linspace(min_x, max_x, int((max_x-min_x)*DENSITY_RES))
    y_range = np.linspace(min_y, max_y, int((max_y-min_y)*DENSITY_RES))
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    candidate_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Check which points are inside the bell shape
    path = Path(polygon_points)
    mask = path.contains_points(candidate_points)
    soft_particles = candidate_points[mask]
    
    # 5. Mirror to create Left Side
    if len(soft_particles) > 0:
        left_particles = soft_particles.copy()
        left_particles[:, 0] *= -1 # Flip X
        all_soft = np.vstack([soft_particles, left_particles])
    else:
        all_soft = np.zeros((0, 2))

    # 6. Generate Payload (Rigid Body)
    # Simple block in the center
    px = np.linspace(-PAYLOAD_WIDTH/2, PAYLOAD_WIDTH/2, int(PAYLOAD_WIDTH*DENSITY_RES))
    py = np.linspace(0, PAYLOAD_HEIGHT, int(PAYLOAD_HEIGHT*DENSITY_RES))
    pgx, pgy = np.meshgrid(px, py)
    payload_particles = np.vstack([pgx.ravel(), pgy.ravel()]).T
    
    # 7. Combine & Apply spawn offset
    # Material ID: 0 = Water (not here), 1 = Jelly (Soft), 2 = Payload (Rigid)
    offset = np.array(spawn_offset)

    final_pos = []
    final_mat = []

    if len(all_soft) > 0:
        final_pos.append(all_soft + offset)
        final_mat.append(np.ones(len(all_soft), dtype=int) * 1)  # Mat 1 = Jelly

    if len(payload_particles) > 0:
        final_pos.append(payload_particles + offset)
        final_mat.append(np.ones(len(payload_particles), dtype=int) * 2)  # Mat 2 = Rigid

    if len(final_pos) > 0:
        return np.vstack(final_pos), np.concatenate(final_mat).astype(int)
    else:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)

def fill_tank(genome, n_particles, spawn_offset=None, water_margin=0.02):
    """
    Creates a complete particle set: robot + background water, padded to fixed size.

    Args:
        genome: 9-element vector defining bell shape
        n_particles: Target total particle count (fixed for GPU tensors)
        spawn_offset: [x, y] position for payload center (default: [0.5, 0.7])
        water_margin: Exclusion radius around robot particles

    Returns:
        positions: (n_particles, 2) array - padded with [-1, -1] for unused slots
        materials: (n_particles,) array - 0=water, 1=jelly, 2=payload, -1=dead
    """
    if spawn_offset is None:
        spawn_offset = DEFAULT_SPAWN

    # 1. Generate robot particles
    robot_pos, robot_mat = generate_phenotype(genome, spawn_offset)
    n_robot = len(robot_pos)

    # 2. Generate water grid covering entire tank [0.05, 0.95] with margin from walls
    tank_margin = 0.05
    water_res = int(np.sqrt(n_particles) * 0.9)  # Approximate grid resolution
    wx = np.linspace(tank_margin, 1.0 - tank_margin, water_res)
    wy = np.linspace(tank_margin, 1.0 - tank_margin, water_res)
    wgx, wgy = np.meshgrid(wx, wy)
    water_candidates = np.vstack([wgx.ravel(), wgy.ravel()]).T

    # 3. Remove water particles that overlap with robot (boolean subtraction)
    if n_robot > 0:
        tree = cKDTree(robot_pos)
        # Find water particles too close to any robot particle
        distances, _ = tree.query(water_candidates, k=1)
        keep_mask = distances > water_margin
        water_pos = water_candidates[keep_mask]
    else:
        water_pos = water_candidates

    n_water = len(water_pos)

    # 4. Combine robot and water
    total_active = n_robot + n_water

    # 5. Allocate fixed-size arrays
    positions = np.full((n_particles, 2), -1.0, dtype=np.float32)
    materials = np.full(n_particles, -1, dtype=np.int32)

    # 6. Fill with robot particles first (they're more important)
    if n_robot > 0:
        positions[:n_robot] = robot_pos
        materials[:n_robot] = robot_mat

    # 7. Fill remaining slots with water (up to capacity)
    water_slots = min(n_water, n_particles - n_robot)
    if water_slots > 0:
        positions[n_robot:n_robot + water_slots] = water_pos[:water_slots]
        materials[n_robot:n_robot + water_slots] = 0  # Mat 0 = Water

    return positions, materials, {
        'n_robot': n_robot,
        'n_water': water_slots,
        'n_total': n_robot + water_slots,
        'n_dead': n_particles - (n_robot + water_slots)
    }


def random_genome():
    """Generate a random but reasonable genome for testing."""
    genome = np.zeros(9)
    genome[0] = np.random.uniform(0.0, 0.15)   # cp1_x positive
    genome[1] = np.random.uniform(-0.05, 0.05) # cp1_y
    genome[2] = np.random.uniform(0.05, 0.2)   # cp2_x
    genome[3] = np.random.uniform(-0.1, 0.05)  # cp2_y
    genome[4] = np.random.uniform(0.1, 0.25)   # end_x (width)
    genome[5] = np.random.uniform(-0.3, -0.1)  # end_y (height/depth, negative = down)
    genome[6] = np.random.uniform(0.02, 0.06)  # t_base
    genome[7] = np.random.uniform(0.02, 0.08)  # t_mid
    genome[8] = np.random.uniform(0.005, 0.02) # t_tip
    return genome


def visualize_population(n_samples=4):
    """Generates N random genomes and displays them."""
    cols = 4
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    print(f"Generating {n_samples} morphologies...")
    
    for i in range(n_samples):
        # Create a random genome
        # Format: [cp1_x, cp1_y, cp2_x, cp2_y, end_x, end_y, t_base, t_mid, t_tip]
        # Using random uniform values to simulate CMA-ES exploration
        genome = np.random.uniform(low=-0.2, high=0.2, size=9)
        
        # Enforce some "reasonable" constraints for the demo
        genome[0] = np.random.uniform(0.0, 0.2)  # cp1_x positive
        genome[1] = np.random.uniform(-0.1, 0.1) # cp1_y
        genome[4] = np.random.uniform(0.1, 0.4)  # end_x (width)
        genome[5] = np.random.uniform(-0.4, -0.1)# end_y (height/depth)
        
        # Thicknesses (must be positive)
        genome[6] = np.random.uniform(0.02, 0.08)
        genome[7] = np.random.uniform(0.02, 0.10)
        genome[8] = np.random.uniform(0.005, 0.03)

        pos, mat = generate_phenotype(genome)
        
        if len(pos) > 0:
            # Plot
            ax = axes[i]
            # Plot Jelly
            jelly = pos[mat == 1]
            if len(jelly) > 0:
                ax.scatter(jelly[:,0], jelly[:,1], s=1, c='cyan', alpha=0.5, label='Soft Body')
            
            # Plot Payload
            payload = pos[mat == 2]
            if len(payload) > 0:
                ax.scatter(payload[:,0], payload[:,1], s=1, c='black', label='Payload')
                
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f"Genotype {i}")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_population(8)