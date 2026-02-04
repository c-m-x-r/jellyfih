"""Configuration dataclasses for MPM simulation parameters."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GridConfig:
    """Grid configuration for MPM simulation."""

    resolution: int = 64  # Grid cells per dimension (64x64)
    dx: float = field(init=False)  # Grid spacing (computed)
    inv_dx: float = field(init=False)  # Inverse grid spacing (computed)

    def __post_init__(self):
        self.dx = 1.0 / self.resolution
        self.inv_dx = float(self.resolution)


@dataclass
class ParticleConfig:
    """Particle configuration for MPM simulation."""

    particles_per_cell: int = 8  # Medium density
    water_region: Tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.5)  # (x_min, y_min, x_max, y_max) normalized

    def compute_particle_count(self, grid_resolution: int) -> int:
        """Compute total particle count based on water region and grid."""
        x_min, y_min, x_max, y_max = self.water_region
        cells_x = int((x_max - x_min) * grid_resolution)
        cells_y = int((y_max - y_min) * grid_resolution)
        return cells_x * cells_y * self.particles_per_cell


@dataclass
class PhysicsConfig:
    """Physics parameters - DiffTaichi style."""

    # Time stepping
    dt: float = 2e-4  # DiffTaichi default timestep
    substeps: int = 50  # Substeps per frame

    # Material properties (Neo-Hookean fluid)
    E: float = 1000.0  # Young's modulus (stiffness)
    nu: float = 0.2  # Poisson's ratio (affects bulk vs shear)
    rho: float = 1.0  # Rest density

    # Derived Lame parameters (computed)
    mu: float = field(init=False)  # Shear modulus
    lambda_: float = field(init=False)  # First Lame parameter (bulk)

    # External forces
    gravity: Tuple[float, float] = (0.0, -9.8)

    def __post_init__(self):
        # Compute Lame parameters from E and nu
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lambda_ = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


@dataclass
class RenderConfig:
    """Rendering configuration for headless output."""

    width: int = 512
    height: int = 512
    fps: int = 30
    particle_radius: int = 2  # Pixel radius for particles

    # Colors (RGB tuples)
    background_color: Tuple[int, int, int] = (20, 20, 30)  # Dark blue-gray
    particle_color: Tuple[int, int, int] = (100, 150, 255)  # Light blue

    # Output
    output_dir: str = "output"
    save_frames: bool = True
    save_gif: bool = True


@dataclass
class SimulationConfig:
    """Master configuration combining all subsystems."""

    grid: GridConfig = field(default_factory=GridConfig)
    particles: ParticleConfig = field(default_factory=ParticleConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    # Simulation duration
    total_frames: int = 150  # 5 seconds at 30 FPS

    @property
    def n_particles(self) -> int:
        return self.particles.compute_particle_count(self.grid.resolution)

    @property
    def steps_per_frame(self) -> int:
        return self.physics.substeps


def default_config() -> SimulationConfig:
    """Create default configuration for water settling demo."""
    return SimulationConfig()


def high_res_config() -> SimulationConfig:
    """Create higher resolution configuration."""
    return SimulationConfig(
        grid=GridConfig(resolution=128),
        particles=ParticleConfig(particles_per_cell=8),
        total_frames=200,
    )
