"""Jellyfish geometry generation.

Creates a half-ellipse dome of particles with the bottom rim
flagged as actuators. Used by the MPM solver for soft body init.
"""

import numpy as np
from typing import Tuple

from .config import JellyfishGeometry


def is_point_in_dome(x: float, y: float,
                     cx: float, cy: float,
                     a: float, b: float) -> bool:
    """Test if (x, y) is inside the upper-half ellipse centered at (cx, cy)."""
    if y < cy:
        return False
    return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0


def is_point_on_rim(x: float, y: float,
                    cx: float, cy: float,
                    a: float, b: float,
                    rim_thickness: float) -> bool:
    """Test if point is in the actuator rim (bottom strip of the dome)."""
    if not is_point_in_dome(x, y, cx, cy, a, b):
        return False
    return y < cy + rim_thickness * b


def compute_jellyfish_particle_count(geometry: JellyfishGeometry, dx: float) -> int:
    """Estimate particle count from half-ellipse area. Includes 1.5x buffer."""
    area = 0.5 * np.pi * geometry.semi_major * geometry.semi_minor
    n_cells = area / (dx * dx)
    return int(n_cells * geometry.particles_per_cell * 1.5)


def generate_jellyfish_particles(
    geometry: JellyfishGeometry,
    dx: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill the dome with jittered particles via rejection sampling.

    Returns:
        positions: (N, 2) float32 array
        is_actuator: (N,) int32 array (1 = rim actuator, 0 = body)
    """
    rng = np.random.default_rng(seed)
    cx, cy = geometry.center
    a, b = geometry.semi_major, geometry.semi_minor
    ppc = geometry.particles_per_cell

    # Bounding box of the dome
    cells_x = int((2 * a) / dx) + 1
    cells_y = int(b / dx) + 1

    positions = []
    actuators = []

    for i in range(cells_x):
        for j in range(cells_y):
            cell_cx = (cx - a) + (i + 0.5) * dx
            cell_cy = cy + (j + 0.5) * dx

            if not is_point_in_dome(cell_cx, cell_cy, cx, cy, a, b):
                continue

            for _ in range(ppc):
                px = cell_cx + (rng.random() - 0.5) * dx * 0.8
                py = cell_cy + (rng.random() - 0.5) * dx * 0.8

                if is_point_in_dome(px, py, cx, cy, a, b):
                    positions.append([px, py])
                    actuators.append(1 if is_point_on_rim(px, py, cx, cy, a, b,
                                                          geometry.rim_thickness) else 0)

    return (np.array(positions, dtype=np.float32),
            np.array(actuators, dtype=np.int32))
