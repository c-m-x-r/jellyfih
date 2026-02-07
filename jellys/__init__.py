"""Jellys: 2D MPM simulation for jellyfish-inspired soft robots."""

from .config import (
    SimulationConfig,
    GridConfig,
    ParticleConfig,
    PhysicsConfig,
    RenderConfig,
    JellyfishConfig,
    JellyfishGeometry,
    JellyfishMaterial,
    JellyfishActuation,
    default_config,
    high_res_config,
    jellyfish_demo_config,
)
from .simulation import MPMSolver, init_taichi, MATERIAL_FLUID, MATERIAL_JELLY
from .rendering import Renderer, render_simulation
from .jellyfish import (
    generate_jellyfish_particles,
    compute_jellyfish_particle_count,
    is_point_in_dome,
    is_point_on_rim,
)

__version__ = "0.3.0"
