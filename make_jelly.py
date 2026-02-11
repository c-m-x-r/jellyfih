import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import cKDTree

# --- CONFIGURATION ---
PAYLOAD_WIDTH = 0.08  
PAYLOAD_HEIGHT = 0.15   
DEFAULT_SPAWN = np.array([0.5, 0.7]) 

def cubic_bezier(p0, p1, p2, p3, t):
    """Returns a point on the cubic Bezier curve at time t."""
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

def get_normals_2d(points):
    """Calculates normal vectors for a 2D line strip."""
    tangents = np.gradient(points, axis=0)
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return normals / norm

def generate_phenotype(genome, spawn_offset=None, grid_res=128):
    """
    Converts a CMA-ES genome vector into particle positions and materials.
    Includes logic for a conditional 'Muscle' layer (Material 3).
    """
    if spawn_offset is None:
        spawn_offset = DEFAULT_SPAWN
    
    # 1. Define Key Points
    start_p = np.array([PAYLOAD_WIDTH / 2.0, 0.0]) 
    
    cp1 = start_p + np.array([abs(genome[0]), genome[1]])
    cp2 = start_p + np.array([abs(genome[2]), genome[3]])
    end_p = start_p + np.array([abs(genome[4]), genome[5]])
    
    t_base = abs(genome[6])
    t_mid = abs(genome[7])
    t_tip = abs(genome[8])
    
    # 2. Generate Spine Curve
    t_steps = np.linspace(0, 1, 50)
    spine_points = np.array([cubic_bezier(start_p, cp1, cp2, end_p, t) for t in t_steps])
    
    # 3. Generate Envelope (Flesh + Muscle)
    normals = get_normals_2d(spine_points)
    t_profile = np.interp(t_steps, [0, 0.5, 1], [t_base, t_mid, t_tip])
    
    # Calculate half-thickness
    semi_thickness = t_profile[:, None] / 2.0
    
    # Outer Curve (Skin)
    outer_curve = spine_points + normals * semi_thickness
    
    # Inner Curve (Cavity)
    inner_curve = spine_points - normals * semi_thickness
    
    # --- MUSCLE LAYER GENERATION (Base + Proportional Scaling) ---
    # We ensure every morphology has at least a 1-particle-thick base layer.
    # This prevents 'dead' phenotypes and provides a smooth evolutionary gradient.
    
    spacing = 1.0 / (grid_res * 2.0)           # Physical particle spacing
    min_base_muscle = spacing * 1.2            # Guaranteed minimum layer (approx 1.2x spacing)
    muscle_ratio = 0.20                        # Proportional thickness (20% of wall)
    
    # Final muscle thickness: Base floor + percentage of local thickness
    # Even at the thinnest tip (t_tip), this will result in a visible muscle layer.
    effective_muscle_thick = min_base_muscle + (semi_thickness * muscle_ratio)
    
    # Ensure the muscle doesn't exceed the total thickness of the bell wall
    effective_muscle_thick = np.minimum(effective_muscle_thick, semi_thickness * 0.9)
    
    # Define the "Muscle Interface" (where Muscle meets Jelly)
    # The muscle lines the interior (inner_curve) and extends into the body.
    muscle_interface = spine_points - normals * (semi_thickness - effective_muscle_thick)
    
    # 4. Define Polygons
    # Body Polygon: The entire solid shape (Outer -> Inner)
    body_polygon = np.vstack([outer_curve, inner_curve[::-1]])
    
    # Muscle Polygon: The active region (Interface -> Inner)
    muscle_polygon = np.vstack([muscle_interface, inner_curve[::-1]])
    
    # 5. Rasterize (Fill Polygon with Particles)
    min_x, min_y = np.min(body_polygon, axis=0)
    max_x, max_y = np.max(body_polygon, axis=0)
    
    # Sync raster resolution with physics grid to prevent clumping
    raster_res = grid_res * 2 
    
    x_range = np.linspace(min_x, max_x, int((max_x-min_x)*raster_res))
    y_range = np.linspace(min_y, max_y, int((max_y-min_y)*raster_res))
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    candidate_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Identify particles inside the Body
    path_body = Path(body_polygon)
    mask_body = path_body.contains_points(candidate_points)
    body_points = candidate_points[mask_body]
    
    # Identify particles inside the Muscle (subset of Body)
    path_muscle = Path(muscle_polygon)
    mask_muscle = path_muscle.contains_points(body_points)
    
    # Assign Materials
    # Default = 1 (Jelly)
    particle_mats = np.ones(len(body_points), dtype=int)
    # Muscle = 3 (Active)
    particle_mats[mask_muscle] = 3
    
    # 6. Mirror to create Left Side
    if len(body_points) > 0:
        left_pos = body_points.copy()
        left_pos[:, 0] *= -1 
        left_mats = particle_mats.copy()
        
        all_soft_pos = np.vstack([body_points, left_pos])
        all_soft_mats = np.concatenate([particle_mats, left_mats])
    else:
        all_soft_pos = np.zeros((0, 2))
        all_soft_mats = np.zeros(0, dtype=int)

    # 7. Generate Payload (Rigid Body - Material 2)
    px = np.linspace(-PAYLOAD_WIDTH/2, PAYLOAD_WIDTH/2, int(PAYLOAD_WIDTH*raster_res))
    py = np.linspace(0, PAYLOAD_HEIGHT, int(PAYLOAD_HEIGHT*raster_res))
    pgx, pgy = np.meshgrid(px, py)
    payload_particles = np.vstack([pgx.ravel(), pgy.ravel()]).T
    
    # 8. Combine & Apply spawn offset
    offset = np.array(spawn_offset)

    final_pos = []
    final_mat = []

    if len(all_soft_pos) > 0:
        final_pos.append(all_soft_pos + offset)
        final_mat.append(all_soft_mats) 

    if len(payload_particles) > 0:
        final_pos.append(payload_particles + offset)
        final_mat.append(np.ones(len(payload_particles), dtype=int) * 2)

    if len(final_pos) > 0:
        return np.vstack(final_pos), np.concatenate(final_mat).astype(int)
    else:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)

def fill_tank(genome, max_particles, grid_res=128, spawn_offset=None, water_margin=0.005):
    """
    Creates a complete particle set with PHYSICALLY CORRECT spacing.
    Args:
        grid_res: The resolution of the physics grid. MUST match mpm_sim.py.
    """
    if spawn_offset is None:
        spawn_offset = DEFAULT_SPAWN

    # 1. Generate robot particles (passing grid_res for correct rasterization)
    robot_pos, robot_mat = generate_phenotype(genome, spawn_offset, grid_res=grid_res)
    n_robot = len(robot_pos)

    # 2. Generate Water Grid
    # Physics requires 2 particles per grid cell (dx/2 spacing).
    spacing = 1.0 / (grid_res * 2.0)
    
    # Create ranges with a half-spacing buffer
    margin = spacing * 3
    wx = np.arange(margin, 1.0 - margin, spacing)
    wy = np.arange(margin, 1.0 - margin, spacing) 

    wgx, wgy = np.meshgrid(wx, wy)
    water_candidates = np.vstack([wgx.ravel(), wgy.ravel()]).T

    # 3. Remove water particles that overlap with robot
    if n_robot > 0:
        tree = cKDTree(robot_pos)
        distances, _ = tree.query(water_candidates, k=1)
        keep_mask = distances > water_margin
        water_pos = water_candidates[keep_mask]
    else:
        water_pos = water_candidates

    n_water = len(water_pos)
    
    # 4. Safety Check
    if n_robot + n_water > max_particles:
        print(f"WARNING: Too many particles! Needed {n_robot + n_water}, but max is {max_particles}.")
        print("Truncating water to fit.")
        n_water = max_particles - n_robot

    # 5. Allocate fixed-size arrays
    positions = np.full((max_particles, 2), -1.0, dtype=np.float32)
    materials = np.full(max_particles, -1, dtype=np.int32)

    # 6. Fill
    if n_robot > 0:
        positions[:n_robot] = robot_pos
        materials[:n_robot] = robot_mat

    if n_water > 0:
        positions[n_robot:n_robot + n_water] = water_pos[:n_water]
        materials[n_robot:n_robot + n_water] = 0 

    return positions, materials, {
        'n_robot': n_robot,
        'n_water': n_water,
        'n_total': n_robot + n_water,
        'n_dead': max_particles - (n_robot + n_water)
    }

def random_genome():
    """Generate a random but reasonable genome for testing."""
    genome = np.zeros(9)
    genome[0] = np.random.uniform(0.0, 0.15)   # cp1_x
    genome[1] = np.random.uniform(-0.05, 0.05) # cp1_y
    genome[2] = np.random.uniform(0.05, 0.2)   # cp2_x
    genome[3] = np.random.uniform(-0.1, 0.05)  # cp2_y
    genome[4] = np.random.uniform(0.1, 0.25)   # end_x
    genome[5] = np.random.uniform(-0.3, -0.1)  # end_y
    genome[6] = np.random.uniform(0.02, 0.06)  # t_base
    genome[7] = np.random.uniform(0.02, 0.08)  # t_mid
    genome[8] = np.random.uniform(0.005, 0.02) # t_tip
    return genome