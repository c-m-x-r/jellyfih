"""Jellyfish geometry generation and particle spawning.

This module handles the creation of jellyfish particles in a half-ellipse
dome shape, with rim particles marked as actuators.
"""

import numpy as np
from typing import Tuple

from .config import JellyfishGeometry


def is_point_in_dome(
    x: float, y: float,
    cx: float, cy: float,
    a: float, b: float
) -> bool:
    """Check if point is inside half-ellipse dome.

    The dome is the upper half of an ellipse centered at (cx, cy).
    Points are inside if they satisfy the ellipse equation AND y >= cy.

    Args:
        x, y: Point coordinates
        cx, cy: Ellipse center (at base of dome)
        a: Semi-major axis (horizontal radius)
        b: Semi-minor axis (vertical height)

    Returns:
        True if point is inside the upper half of the ellipse
    """
    if y < cy:
        return False

    # Ellipse equation: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
    dx = (x - cx) / a
    dy = (y - cy) / b
    return (dx * dx + dy * dy) <= 1.0


def is_point_on_rim(
    x: float, y: float,
    cx: float, cy: float,
    a: float, b: float,
    rim_thickness: float
) -> bool:
    """Check if point is in the rim region (bottom edge of dome).

    The rim is defined as points inside the dome where:
    y < cy + rim_thickness * b

    Args:
        x, y: Point coordinates
        cx, cy: Ellipse center
        a, b: Semi-axes
        rim_thickness: Fraction of b that defines rim height (0.0-1.0)

    Returns:
        True if point is in the actuator rim region
    """
    if not is_point_in_dome(x, y, cx, cy, a, b):
        return False

    rim_y_threshold = cy + rim_thickness * b
    return y < rim_y_threshold


def compute_jellyfish_particle_count(geometry: JellyfishGeometry, dx: float) -> int:
    """Estimate number of particles needed for jellyfish geometry.

    Uses the area of a half-ellipse: A = (pi * a * b) / 2

    Args:
        geometry: Jellyfish geometry configuration
        dx: Grid cell size

    Returns:
        Approximate particle count for the jellyfish
    """
    a = geometry.semi_major
    b = geometry.semi_minor
    ppc = geometry.particles_per_cell

    # Half-ellipse area
    area = 0.5 * np.pi * a * b

    # Particles based on cell area coverage
    cell_area = dx * dx
    n_cells = area / cell_area

    return int(n_cells * ppc * 1.2)  # 20% buffer for jitter


def generate_jellyfish_particles(
    geometry: JellyfishGeometry,
    dx: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate particle positions and rim flags for jellyfish.

    Creates a half-ellipse dome filled with particles using rejection sampling.
    Bottom rim particles are marked as actuators.

    Args:
        geometry: Jellyfish geometry configuration
        dx: Grid cell size for sampling density
        seed: Random seed for reproducible jitter

    Returns:
        positions: (N, 2) array of particle positions
        is_actuator: (N,) int array, 1 for rim particles, 0 otherwise
    """
    rng = np.random.default_rng(seed)

    cx, cy = geometry.center
    a = geometry.semi_major
    b = geometry.semi_minor
    rim_thickness = geometry.rim_thickness
    ppc = geometry.particles_per_cell

    # Sampling grid bounds (bounding box of dome)
    x_min = cx - a
    x_max = cx + a
    y_min = cy
    y_max = cy + b

    # Sample cells within bounding box
    cells_x = int((x_max - x_min) / dx) + 1
    cells_y = int((y_max - y_min) / dx) + 1

    positions_list = []
    actuator_list = []

    for i in range(cells_x):
        for j in range(cells_y):
            cell_cx = x_min + (i + 0.5) * dx
            cell_cy = y_min + (j + 0.5) * dx

            # Check if cell center is inside dome
            if not is_point_in_dome(cell_cx, cell_cy, cx, cy, a, b):
                continue

            # Spawn particles in this cell
            for _ in range(ppc):
                jitter_x = (rng.random() - 0.5) * dx * 0.8
                jitter_y = (rng.random() - 0.5) * dx * 0.8

                px = cell_cx + jitter_x
                py = cell_cy + jitter_y

                # Verify jittered point is still inside dome
                if is_point_in_dome(px, py, cx, cy, a, b):
                    positions_list.append([px, py])

                    # Check if this is a rim (actuator) particle
                    is_rim = is_point_on_rim(px, py, cx, cy, a, b, rim_thickness)
                    actuator_list.append(1 if is_rim else 0)

    positions = np.array(positions_list, dtype=np.float32)
    is_actuator = np.array(actuator_list, dtype=np.int32)

    return positions, is_actuator


def get_jellyfish_center_of_mass(positions: np.ndarray) -> Tuple[float, float]:
    """Compute center of mass of jellyfish particles.

    Useful for tracking propulsion during simulation.

    Args:
        positions: (N, 2) array of particle positions

    Returns:
        (x, y) center of mass coordinates
    """
    if len(positions) == 0:
        return (0.0, 0.0)

    return (float(positions[:, 0].mean()), float(positions[:, 1].mean()))
