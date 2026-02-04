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
    water_region: Tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.7)  # 70% water level

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

    # Material properties (fluid)
    E: float = 1000.0  # Young's modulus (stiffness)
    nu: float = 0.2  # Poisson's ratio (affects bulk vs shear)
    rho: float = 1.0  # Rest density

    # Derived Lame parameters (computed)
    mu: float = field(init=False)  # Shear modulus
    lambda_: float = field(init=False)  # First Lame parameter (bulk)

    # External forces
    gravity: Tuple[float, float] = (0.0, -9.8)

    # Water level (y-coordinate of surface)
    water_level: float = 0.7

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
    particle_color: Tuple[int, int, int] = (100, 150, 255)  # Light blue (fluid)
    jelly_color: Tuple[int, int, int] = (255, 150, 200)  # Pink (jellyfish body)
    actuator_color: Tuple[int, int, int] = (255, 100, 150)  # Bright pink (rim)

    # Water surface visualization
    draw_water_level: bool = True
    water_level_color: Tuple[int, int, int] = (80, 120, 200)

    # Output
    output_dir: str = "output"
    save_frames: bool = True
    save_gif: bool = True


# =============================================================================
# Jellyfish Configuration
# =============================================================================


@dataclass
class JellyfishGeometry:
    """Jellyfish bell geometry parameters."""

    # Spawn position (center of dome base)
    center: Tuple[float, float] = (0.5, 0.5)

    # Half-ellipse dimensions
    semi_major: float = 0.12  # Horizontal radius
    semi_minor: float = 0.08  # Vertical height (dome)

    # Rim region (actuators)
    rim_thickness: float = 0.2  # Fraction of semi_minor defining rim height

    # Sampling density
    particles_per_cell: int = 8


@dataclass
class JellyfishMaterial:
    """Jellyfish material properties."""

    # Constitutive model: "fixed_corotated" or "neo_hookean"
    model: str = "fixed_corotated"

    # Elastic properties
    E: float = 3000.0  # Young's modulus (medium stiffness)
    nu: float = 0.3  # Poisson's ratio

    # Density relative to water (1.0 = neutral buoyancy)
    density: float = 1.0

    # Derived Lame parameters (computed)
    mu: float = field(init=False)
    lambda_: float = field(init=False)

    def __post_init__(self):
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lambda_ = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


@dataclass
class JellyfishActuation:
    """Actuation parameters for jellyfish contraction."""

    frequency: float = 1.5  # Hz (contractions per second)
    amplitude: float = 0.3  # Contraction strength (0.0-1.0)
    phase_offset: float = 0.0  # Initial phase in radians


@dataclass
class JellyfishConfig:
    """Complete jellyfish configuration."""

    enabled: bool = False
    geometry: JellyfishGeometry = field(default_factory=JellyfishGeometry)
    material: JellyfishMaterial = field(default_factory=JellyfishMaterial)
    actuation: JellyfishActuation = field(default_factory=JellyfishActuation)


# =============================================================================
# Master Configuration
# =============================================================================


@dataclass
class SimulationConfig:
    """Master configuration combining all subsystems."""

    grid: GridConfig = field(default_factory=GridConfig)
    particles: ParticleConfig = field(default_factory=ParticleConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    jellyfish: JellyfishConfig = field(default_factory=JellyfishConfig)

    # Simulation duration
    total_frames: int = 150  # 5 seconds at 30 FPS

    @property
    def n_particles(self) -> int:
        """Total particle count (water only, jellyfish added separately)."""
        return self.particles.compute_particle_count(self.grid.resolution)

    @property
    def steps_per_frame(self) -> int:
        return self.physics.substeps


# =============================================================================
# Factory Functions
# =============================================================================


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


def jellyfish_demo_config() -> SimulationConfig:
    """Create configuration for jellyfish swimming demonstration.

    - 70% water level
    - Jellyfish spawned fully submerged at center
    - ~10 contraction cycles (at 1.5 Hz over ~7 seconds)
    """
    return SimulationConfig(
        grid=GridConfig(resolution=64),
        particles=ParticleConfig(
            particles_per_cell=8,
            water_region=(0.1, 0.1, 0.9, 0.7),  # 70% water level
        ),
        physics=PhysicsConfig(
            water_level=0.7,
        ),
        jellyfish=JellyfishConfig(
            enabled=True,
            geometry=JellyfishGeometry(
                center=(0.5, 0.45),  # Submerged (below water level 0.7)
                semi_major=0.12,
                semi_minor=0.08,
            ),
            material=JellyfishMaterial(
                model="fixed_corotated",
                E=3000.0,
                density=1.0,  # Neutral buoyancy
            ),
            actuation=JellyfishActuation(
                frequency=1.5,
                amplitude=0.3,
            ),
        ),
        total_frames=210,  # ~7 seconds at 30 FPS = ~10 cycles
    )
