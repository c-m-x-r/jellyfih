"""Configuration dataclasses for MPM jellyfish simulation.

All spatial quantities are in normalized [0, 1] simulation coordinates.
"""

from dataclasses import dataclass, field
from typing import Tuple


# =============================================================================
# Grid & Particles
# =============================================================================


@dataclass
class GridConfig:
    resolution: int = 64  # NxN grid cells

    # Derived (computed from resolution)
    dx: float = field(init=False)
    inv_dx: float = field(init=False)

    def __post_init__(self):
        self.dx = 1.0 / self.resolution
        self.inv_dx = float(self.resolution)


@dataclass
class ParticleConfig:
    particles_per_cell: int = 8
    water_region: Tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.7)

    def compute_particle_count(self, grid_resolution: int) -> int:
        x_min, y_min, x_max, y_max = self.water_region
        cells_x = int((x_max - x_min) * grid_resolution)
        cells_y = int((y_max - y_min) * grid_resolution)
        return cells_x * cells_y * self.particles_per_cell


# =============================================================================
# Physics
# =============================================================================


@dataclass
class PhysicsConfig:
    dt: float = 2e-4
    substeps: int = 50  # substeps per rendered frame

    # Fluid material (weakly compressible)
    E: float = 1000.0   # Young's modulus
    nu: float = 0.2      # Poisson's ratio

    gravity: Tuple[float, float] = (0.0, -9.8)
    water_level: float = 0.7  # y-coordinate of initial water surface

    # Derived Lame parameters
    mu: float = field(init=False)
    lambda_: float = field(init=False)

    def __post_init__(self):
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lambda_ = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


# =============================================================================
# Rendering
# =============================================================================


@dataclass
class RenderConfig:
    width: int = 512
    height: int = 512
    fps: int = 30
    particle_radius: int = 2

    background_color: Tuple[int, int, int] = (20, 20, 30)
    particle_color: Tuple[int, int, int] = (100, 150, 255)    # fluid
    jelly_color: Tuple[int, int, int] = (255, 150, 200)       # bell body
    actuator_color: Tuple[int, int, int] = (255, 100, 150)    # actuator rim

    draw_water_level: bool = True
    water_level_color: Tuple[int, int, int] = (80, 120, 200)

    output_dir: str = "output"
    save_frames: bool = True
    save_gif: bool = True


# =============================================================================
# Jellyfish
# =============================================================================


@dataclass
class JellyfishGeometry:
    center: Tuple[float, float] = (0.5, 0.5)     # dome base center
    semi_major: float = 0.12                       # horizontal radius
    semi_minor: float = 0.08                       # vertical dome height
    rim_thickness: float = 0.2                     # fraction of semi_minor that is actuator
    particles_per_cell: int = 8


@dataclass
class JellyfishMaterial:
    model: str = "fixed_corotated"  # or "neo_hookean"
    E: float = 3000.0               # Young's modulus
    nu: float = 0.3                 # Poisson's ratio
    density: float = 1.0            # relative to water (1.0 = neutral buoyancy)

    # Derived Lame parameters
    mu: float = field(init=False)
    lambda_: float = field(init=False)

    def __post_init__(self):
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lambda_ = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


@dataclass
class JellyfishActuation:
    frequency: float = 1.5    # Hz
    amplitude: float = 0.3    # contraction strength [0, 1]
    phase_offset: float = 0.0 # radians


@dataclass
class JellyfishConfig:
    enabled: bool = False
    geometry: JellyfishGeometry = field(default_factory=JellyfishGeometry)
    material: JellyfishMaterial = field(default_factory=JellyfishMaterial)
    actuation: JellyfishActuation = field(default_factory=JellyfishActuation)


# =============================================================================
# Master Config
# =============================================================================


@dataclass
class SimulationConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    particles: ParticleConfig = field(default_factory=ParticleConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    jellyfish: JellyfishConfig = field(default_factory=JellyfishConfig)
    total_frames: int = 150  # ~5 seconds at 30 FPS

    @property
    def n_particles(self) -> int:
        """Water particle count (jellyfish particles added separately)."""
        return self.particles.compute_particle_count(self.grid.resolution)

    @property
    def steps_per_frame(self) -> int:
        return self.physics.substeps


# =============================================================================
# Presets
# =============================================================================


def default_config() -> SimulationConfig:
    return SimulationConfig()


def high_res_config() -> SimulationConfig:
    return SimulationConfig(
        grid=GridConfig(resolution=128),
        particles=ParticleConfig(particles_per_cell=8),
        total_frames=200,
    )


def jellyfish_demo_config() -> SimulationConfig:
    """Jellyfish submerged in 70% water, ~10 contraction cycles."""
    return SimulationConfig(
        grid=GridConfig(resolution=64),
        particles=ParticleConfig(
            particles_per_cell=8,
            water_region=(0.1, 0.1, 0.9, 0.7),
        ),
        physics=PhysicsConfig(water_level=0.7),
        jellyfish=JellyfishConfig(
            enabled=True,
            geometry=JellyfishGeometry(
                center=(0.5, 0.45),
                semi_major=0.12,
                semi_minor=0.08,
            ),
            material=JellyfishMaterial(
                model="fixed_corotated",
                E=3000.0,
                density=1.0,
            ),
            actuation=JellyfishActuation(
                frequency=1.5,
                amplitude=0.3,
            ),
        ),
        total_frames=210,  # ~7 seconds at 30 FPS
    )
