import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import cKDTree

# --- CONFIGURATION ---
PAYLOAD_WIDTH = 0.08
PAYLOAD_HEIGHT = 0.05
DEFAULT_SPAWN = np.array([0.5, 0.7])

# Aurelia aurita (moon jelly) reference genome:
# Wide, shallow bell with moderate thickness — biomimetic baseline.
# Approximates the saucer-shaped medusa with thin margin.
AURELIA_GENOME = np.array([
    0.05,   # cp1_x: gentle outward curve near payload
    0.04,   # cp1_y: slight upward bulge (dome apex)
    0.18,   # cp2_x: wide mid-bell
    -0.03,  # cp2_y: gentle downward sweep
    0.22,   # end_x: wide bell margin
    -0.12,  # end_y: shallow depth (saucer shape, not deep cone)
    0.04,   # t_base: moderate base thickness
    0.05,   # t_mid: thick mesoglea mid-bell
    0.015,  # t_tip: thin bell margin
]) 

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
    Includes a 'Mesoglea Collar' and 'Transverse Bridge' to unify the body.
    """
    if spawn_offset is None:
        spawn_offset = DEFAULT_SPAWN
    
    # 1. Define Key Points
    # The bell starts at the bottom corner of the payload
    start_p = np.array([PAYLOAD_WIDTH / 2.0, 0.0]) 
    
    # Genes 0-5: Shape Control (Bezier Control Points)
    cp1 = start_p + np.array([abs(genome[0]), genome[1]])
    cp2 = start_p + np.array([abs(genome[2]), genome[3]])
    end_p = start_p + np.array([abs(genome[4]), genome[5]])
    
    # Genes 6-8: Thickness Control
    t_base = abs(genome[6])
    t_mid = abs(genome[7])
    t_tip = abs(genome[8])
    
    # 2. Generate Spine Curve
    t_steps = np.linspace(0, 1, 50)
    spine_points = np.array([cubic_bezier(start_p, cp1, cp2, end_p, t) for t in t_steps])
    
    # 3. Generate Envelope (Flesh + Muscle)
    normals = get_normals_2d(spine_points)
    t_profile = np.interp(t_steps, [0, 0.5, 1], [t_base, t_mid, t_tip])
    semi_thickness = t_profile[:, None] / 2.0
    
    outer_curve = spine_points + normals * semi_thickness
    inner_curve = spine_points - normals * semi_thickness
    
    # --- MUSCLE LAYER GENERATION ---
    spacing = 1.0 / (grid_res * 2.0)
    min_base_muscle = spacing * 1.2
    muscle_ratio = 0.25  # Muscle is 25% of wall thickness
    
    effective_muscle_thick = min_base_muscle + (semi_thickness * muscle_ratio)
    effective_muscle_thick = np.minimum(effective_muscle_thick, semi_thickness * 0.9)
    muscle_interface = spine_points - normals * (semi_thickness - effective_muscle_thick)
    
    # 4. Define Polygons
    body_polygon = np.vstack([outer_curve, inner_curve[::-1]])
    muscle_polygon = np.vstack([muscle_interface, inner_curve[::-1]])
    
    # --- RASTERIZATION SETUP ---
    raster_res = grid_res * 2 
    
    # Bounding box setup (Extra padding for collar/bridge)
    min_x = min(np.min(body_polygon[:, 0]), -PAYLOAD_WIDTH) 
    max_x = max(np.max(body_polygon[:, 0]), PAYLOAD_WIDTH)
    
    # Extend Y range downwards for the bridge and upwards for payload
    min_y = min(np.min(body_polygon[:, 1]), -0.05) 
    max_y = max(np.max(body_polygon[:, 1]), PAYLOAD_HEIGHT)
    
    PAD = 0.05
    x_range = np.linspace(min_x - PAD, max_x + PAD, int((max_x - min_x + 2*PAD)*raster_res))
    y_range = np.linspace(min_y - PAD, max_y + PAD, int((max_y - min_y + 2*PAD)*raster_res))
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    candidate_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # 5. Define Structural Masks
    
    # A. The Bell (Morphology)
    path_body = Path(body_polygon)
    mask_bell = path_body.contains_points(candidate_points)
    
    # B. The Muscle (Active Tissue)
    path_muscle = Path(muscle_polygon)
    mask_muscle = path_muscle.contains_points(candidate_points)
    
    # C. The Collar (Socket around Payload)
    # Wraps the bottom 40% of the payload sides
    collar_w = PAYLOAD_WIDTH + 0.03
    collar_h = PAYLOAD_HEIGHT * 0.40
    
    mask_collar_box = (
        (candidate_points[:, 0] >= 0) & 
        (candidate_points[:, 0] <= collar_w/2.0) &
        (candidate_points[:, 1] >= 0) &
        (candidate_points[:, 1] <= collar_h)
    )
    # Exclude the actual payload space so we don't overlap rigid body
    mask_payload_space_positive = (
        (candidate_points[:, 0] <= PAYLOAD_WIDTH/2.0) &
        (candidate_points[:, 1] <= PAYLOAD_HEIGHT)
    )
    mask_collar = mask_collar_box & ~mask_payload_space_positive
    
    # D. The Transverse Bridge (NEW)
    # A plate of jelly UNDER the payload connecting left and right sides.
    # Spans full width, centered at x=0.
    # Y-range: -0.03 to 0.0 (Below payload)
    bridge_thick = 0.03
    mask_bridge = (
        (candidate_points[:, 0] >= -collar_w/2.0) & # Connects to outer edge of collar
        (candidate_points[:, 0] <= collar_w/2.0) &
        (candidate_points[:, 1] >= -bridge_thick) &
        (candidate_points[:, 1] <= 0.0)
    )
    
    # 6. Assemble Soft Body Materials
    final_mats = np.zeros(len(candidate_points), dtype=int)
    
    # Order matters (later overwrites earlier):
    final_mats[mask_bridge] = 1 # Bridge (Jelly)
    final_mats[mask_collar] = 1 # Collar (Jelly)
    final_mats[mask_bell] = 1   # Bell (Jelly)
    final_mats[mask_bell & mask_muscle] = 3 # Muscle (Active)
    
    # Extract right-side particles (Bridge is centered, so we keep x>0 for mirror logic, 
    # OR we handle bridge separately. Easier to let mirror handle x>0 and add bridge center explicitly)
    
    # Current candidates cover the whole area. 
    # Let's split into Right-Side logic to match existing mirror workflow.
    
    # Filter for Right Side (x >= 0) OR Bridge (x covers 0)
    # Actually, simpler approach: Just define the WHOLE soft body (Bridge + Right) 
    # then mirror the Right parts.
    
    # Extract Right-Side Bell/Collar
    mask_right_side = (candidate_points[:, 0] >= 0) & (final_mats > 0)
    right_points = candidate_points[mask_right_side]
    right_mats = final_mats[mask_right_side]
    
    # 7. Mirror to create Full Body
    if len(right_points) > 0:
        left_pos = right_points.copy()
        left_pos[:, 0] *= -1 
        left_mats = right_mats.copy()
        
        # Combine
        all_soft_pos = np.vstack([right_points, left_pos])
        all_soft_mats = np.concatenate([right_mats, left_mats])
        
        # Add the Central Bridge Slice (strip near x=0 that might get missed by mirror gap)
        # Or better: The Bridge mask above covers -W to W. 
        # But we only extracted x>=0. Mirroring x>=0 covers x<=0. 
        # This works perfectly.
    else:
        all_soft_pos = np.zeros((0, 2))
        all_soft_mats = np.zeros(0, dtype=int)

    # 8. Generate Payload (Material 2)
    # Explicitly generated to ensure it fits perfectly in the gap we left
    px = np.linspace(-PAYLOAD_WIDTH/2, PAYLOAD_WIDTH/2, int(PAYLOAD_WIDTH*raster_res))
    py = np.linspace(0, PAYLOAD_HEIGHT, int(PAYLOAD_HEIGHT*raster_res))
    pgx, pgy = np.meshgrid(px, py)
    payload_particles = np.vstack([pgx.ravel(), pgy.ravel()]).T
    
    # 9. Combine & Apply spawn offset
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
    """
    if spawn_offset is None:
        spawn_offset = DEFAULT_SPAWN

    # 1. Generate robot particles
    robot_pos, robot_mat = generate_phenotype(genome, spawn_offset, grid_res=grid_res)
    n_robot = len(robot_pos)

    # 2. Generate Water Grid
    spacing = 1.0 / (grid_res * 2.0)
    
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

    muscle_count = int(np.sum(materials[:n_robot] == 3))

    return positions, materials, {
        'n_robot': n_robot,
        'n_water': n_water,
        'n_total': n_robot + n_water,
        'n_dead': max_particles - (n_robot + n_water),
        'muscle_count': muscle_count,
    }

def random_genome():
    """Generate a random but reasonable genome for testing."""
    # 9 Genes: Shape (6) + Thickness (3)
    # Bounds match GENOME_LOWER/GENOME_UPPER in evolve.py
    genome = np.zeros(9)
    genome[0] = np.random.uniform(0.0, 0.25)    # cp1_x
    genome[1] = np.random.uniform(-0.15, 0.15)  # cp1_y
    genome[2] = np.random.uniform(0.0, 0.3)     # cp2_x
    genome[3] = np.random.uniform(-0.2, 0.15)   # cp2_y
    genome[4] = np.random.uniform(0.05, 0.35)   # end_x
    genome[5] = np.random.uniform(-0.45, -0.03) # end_y
    genome[6] = np.random.uniform(0.025, 0.08)  # t_base
    genome[7] = np.random.uniform(0.025, 0.1)   # t_mid
    genome[8] = np.random.uniform(0.01, 0.04)   # t_tip
    return genome


if __name__ == "__main__":
    import sys

    # Show Aurelia genome if --aurelia flag, otherwise random
    if "--aurelia" in sys.argv:
        genome = AURELIA_GENOME
        title = "Aurelia aurita (Moon Jelly) Reference"
    else:
        genome = random_genome()
        title = "Random Genome"

    pos, mat = generate_phenotype(genome)
    print(f"{title}: {genome}")
    print(f"Particles: {len(pos)} (jelly={np.sum(mat==1)}, muscle={np.sum(mat==3)}, payload={np.sum(mat==2)})")

    colors = {0: '#4488cc', 1: '#44cc88', 2: '#ff6633', 3: '#ffcc00'}
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for m_id, label in [(1, 'Jelly'), (3, 'Muscle'), (2, 'Payload')]:
        mask = mat == m_id
        if np.any(mask):
            ax.scatter(pos[mask, 0], pos[mask, 1], c=colors[m_id],
                       s=1, label=label, alpha=0.8)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/morphology_preview.png", dpi=150)
    print("Saved to output/morphology_preview.png")
    plt.show()