"""Jellys: 2D Differentiable MPM Simulation for Jellyfish-Inspired Soft Robots.

Phase 1: The Differentiable Aquarium - Stable 2D MPM fluid simulation.
"""

from .config import (
    SimulationConfig,
    GridConfig,
    ParticleConfig,
    PhysicsConfig,
    RenderConfig,
    default_config,
    high_res_config,
)
from .simulation import MPMSolver, init_taichi
from .rendering import Renderer, render_simulation

__version__ = "0.1.0"
__all__ = [
    "SimulationConfig",
    "GridConfig",
    "ParticleConfig",
    "PhysicsConfig",
    "RenderConfig",
    "default_config",
    "high_res_config",
    "MPMSolver",
    "init_taichi",
    "Renderer",
    "render_simulation",
]
