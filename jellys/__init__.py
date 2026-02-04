"""Jellys: 2D Differentiable MPM Simulation for Jellyfish-Inspired Soft Robots.

Phase 1: The Differentiable Aquarium - Stable 2D MPM fluid simulation.
Phase 2: The Parametric Jellyfish - Soft body morphology and actuation.
"""

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
    get_jellyfish_center_of_mass,
)

__version__ = "0.2.0"
__all__ = [
    # Config
    "SimulationConfig",
    "GridConfig",
    "ParticleConfig",
    "PhysicsConfig",
    "RenderConfig",
    "JellyfishConfig",
    "JellyfishGeometry",
    "JellyfishMaterial",
    "JellyfishActuation",
    "default_config",
    "high_res_config",
    "jellyfish_demo_config",
    # Simulation
    "MPMSolver",
    "init_taichi",
    "MATERIAL_FLUID",
    "MATERIAL_JELLY",
    # Rendering
    "Renderer",
    "render_simulation",
    # Jellyfish geometry
    "generate_jellyfish_particles",
    "compute_jellyfish_particle_count",
    "is_point_in_dome",
    "is_point_on_rim",
    "get_jellyfish_center_of_mass",
]
