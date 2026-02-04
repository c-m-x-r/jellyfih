"""2D MPM Solver using Taichi - DiffTaichi style.

This module implements a Material Point Method solver for fluid simulation,
following the DiffTaichi (Hu et al., ICLR 2020) patterns.

Architecture is designed to support future differentiability.
"""

import taichi as ti
import numpy as np
from typing import Optional

from .config import SimulationConfig, default_config


@ti.data_oriented
class MPMSolver:
    """2D MPM Solver for weakly compressible fluid (Neo-Hookean style).

    Follows DiffTaichi architecture for future gradient support.
    """

    def __init__(self, config: Optional[SimulationConfig] = None, seed: int = 42):
        """Initialize the MPM solver.

        Args:
            config: Simulation configuration (uses default if None)
            seed: Random seed for reproducible particle placement
        """
        self.config = config or default_config()
        self.seed = seed
        self._init_fields()
        self._init_particles()

    def _init_fields(self) -> None:
        """Initialize Taichi fields for particles and grid."""
        cfg = self.config
        n_particles = cfg.n_particles
        n_grid = cfg.grid.resolution

        # Particle fields - position and velocity stored for potential gradient tracking
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # position
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # velocity
        self.C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # affine velocity matrix (APIC)
        self.F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # deformation gradient
        self.J = ti.field(dtype=ti.f32, shape=n_particles)  # determinant of F (volume ratio)

        # Grid fields
        self.grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))  # grid velocity
        self.grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))  # grid mass

        # Store config values as Taichi-friendly constants
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.dx = cfg.grid.dx
        self.inv_dx = cfg.grid.inv_dx
        self.dt = cfg.physics.dt
        self.p_mass = 1.0  # particle mass (uniform)

        # Particle volume: total cell volume / particles per cell
        # This ensures mass conservation
        self.p_vol = (self.dx ** 2) / cfg.particles.particles_per_cell

        # Bulk modulus for weakly compressible fluid (2D plane strain)
        # K = E / (2 * (1 - nu)) for 2D
        E = cfg.physics.E
        nu = cfg.physics.nu
        self.bulk_modulus = E / (2.0 * (1.0 - nu))

        # Gravity
        self.gravity = ti.Vector(cfg.physics.gravity)

    def _init_particles(self) -> None:
        """Initialize particle positions in water region."""
        cfg = self.config
        x_min, y_min, x_max, y_max = cfg.particles.water_region
        n_particles = cfg.n_particles

        # Set random seed for reproducibility
        rng = np.random.default_rng(self.seed)

        # Generate particle positions on CPU, then copy to GPU
        positions = np.zeros((n_particles, 2), dtype=np.float32)

        # Calculate grid of particles
        cells_x = int((x_max - x_min) * cfg.grid.resolution)
        cells_y = int((y_max - y_min) * cfg.grid.resolution)
        ppc = cfg.particles.particles_per_cell

        idx = 0
        for i in range(cells_x):
            for j in range(cells_y):
                cell_x = x_min + (i + 0.5) * self.dx
                cell_y = y_min + (j + 0.5) * self.dx
                # Jitter particles within cell
                for _ in range(ppc):
                    if idx < n_particles:
                        jitter_x = (rng.random() - 0.5) * self.dx * 0.8
                        jitter_y = (rng.random() - 0.5) * self.dx * 0.8
                        positions[idx] = [cell_x + jitter_x, cell_y + jitter_y]
                        idx += 1

        # Copy to Taichi fields
        self.x.from_numpy(positions)

        # Initialize other fields
        self._reset_particle_state()

    @ti.kernel
    def _reset_particle_state(self):
        """Reset velocity, affine matrix, and deformation gradient."""
        for p in range(self.n_particles):
            self.v[p] = ti.Vector([0.0, 0.0])
            self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            self.F[p] = ti.Matrix.identity(ti.f32, 2)
            self.J[p] = 1.0

    @ti.kernel
    def _clear_grid(self):
        """Reset grid velocity and mass to zero."""
        for i, j in self.grid_v:
            self.grid_v[i, j] = ti.Vector([0.0, 0.0])
            self.grid_m[i, j] = 0.0

    @ti.kernel
    def _p2g(self):
        """Particle-to-Grid transfer using quadratic B-spline weights.

        Transfers particle momentum to grid nodes.
        Computes stress from weakly compressible fluid model.
        """
        for p in range(self.n_particles):
            # Get particle position in grid coordinates
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)  # Base grid node
            fx = Xp - ti.cast(base, ti.f32)  # Fractional position within cell

            # Quadratic B-spline weights (3x3 stencil)
            # Computed separately for x and y dimensions for clarity
            wx = ti.Vector([
                0.5 * (1.5 - fx.x) ** 2,
                0.75 - (fx.x - 1.0) ** 2,
                0.5 * (fx.x - 0.5) ** 2,
            ])
            wy = ti.Vector([
                0.5 * (1.5 - fx.y) ** 2,
                0.75 - (fx.y - 1.0) ** 2,
                0.5 * (fx.y - 0.5) ** 2,
            ])

            # Weakly compressible fluid stress computation
            J = self.J[p]

            # Pressure from equation of state: p = K * (J - 1) / J
            # Division by J converts to Cauchy stress
            # Negative sign: positive pressure = compression
            pressure = -self.bulk_modulus * (J - 1.0) / J

            # Cauchy stress (isotropic for fluid)
            stress = ti.Matrix([
                [pressure, 0.0],
                [0.0, pressure]
            ])

            # APIC momentum transfer: stress contribution + affine velocity
            affine = stress * self.p_vol * self.inv_dx + self.p_mass * self.C[p]

            # Scatter to 3x3 grid neighborhood
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = wx[i] * wy[j]

                grid_idx = base + offset
                if 0 <= grid_idx.x < self.n_grid and 0 <= grid_idx.y < self.n_grid:
                    # Accumulate momentum and mass (atomic on GPU)
                    self.grid_v[grid_idx] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                    self.grid_m[grid_idx] += weight * self.p_mass

    @ti.kernel
    def _grid_op(self):
        """Grid operations: normalize velocity, apply gravity, enforce boundaries."""
        for i, j in self.grid_v:
            if self.grid_m[i, j] > 1e-10:
                # Normalize momentum to get velocity
                self.grid_v[i, j] /= self.grid_m[i, j]

                # Apply gravity
                self.grid_v[i, j] += self.dt * self.gravity

                # Boundary conditions (sticky/separating at walls)
                bound = 3  # boundary thickness in cells
                if i < bound and self.grid_v[i, j].x < 0:
                    self.grid_v[i, j].x = 0
                if i >= self.n_grid - bound and self.grid_v[i, j].x > 0:
                    self.grid_v[i, j].x = 0
                if j < bound and self.grid_v[i, j].y < 0:
                    self.grid_v[i, j].y = 0
                if j >= self.n_grid - bound and self.grid_v[i, j].y > 0:
                    self.grid_v[i, j].y = 0

    @ti.kernel
    def _g2p(self):
        """Grid-to-Particle transfer.

        Interpolates grid velocities to particles and updates positions.
        Updates deformation gradient F and volume ratio J.
        """
        for p in range(self.n_particles):
            # Get particle position in grid coordinates
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)

            # Quadratic B-spline weights (separate x and y)
            wx = ti.Vector([
                0.5 * (1.5 - fx.x) ** 2,
                0.75 - (fx.x - 1.0) ** 2,
                0.5 * (fx.x - 0.5) ** 2,
            ])
            wy = ti.Vector([
                0.5 * (1.5 - fx.y) ** 2,
                0.75 - (fx.y - 1.0) ** 2,
                0.5 * (fx.y - 0.5) ** 2,
            ])

            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix.zero(ti.f32, 2, 2)

            # Gather from 3x3 grid neighborhood
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = ti.cast(offset, ti.f32) - fx  # Unscaled for APIC
                weight = wx[i] * wy[j]

                grid_idx = base + offset
                if 0 <= grid_idx.x < self.n_grid and 0 <= grid_idx.y < self.n_grid:
                    g_v = self.grid_v[grid_idx]
                    new_v += weight * g_v
                    # APIC: reconstruct affine velocity field
                    # C = B @ D^-1, where D^-1 = 4 * inv_dx^2 for quadratic B-spline
                    new_C += 4.0 * self.inv_dx * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v[p] = new_v
            self.C[p] = new_C

            # Update position
            self.x[p] += self.dt * new_v

            # Update deformation gradient F
            # F_new = (I + dt * grad_v) @ F_old
            # For APIC with quadratic B-spline: grad_v = C (since we scaled C by D^-1)
            F_new = (ti.Matrix.identity(ti.f32, 2) + self.dt * new_C) @ self.F[p]
            self.F[p] = F_new

            # Update J (determinant of F) - volume ratio
            self.J[p] = F_new.determinant()

            # Clamp J to prevent instability (both lower and upper bounds)
            self.J[p] = ti.max(ti.min(self.J[p], 4.0), 0.1)

            # For fluid: reset F to isotropic while preserving J
            # This removes shear deformation history (fluids don't resist shear)
            sqrt_J = ti.sqrt(self.J[p])
            self.F[p] = ti.Matrix([[sqrt_J, 0.0], [0.0, sqrt_J]])

    def substep(self) -> None:
        """Execute one simulation substep: P2G -> Grid ops -> G2P."""
        self._clear_grid()
        self._p2g()
        self._grid_op()
        self._g2p()

    def step(self) -> None:
        """Execute one frame's worth of substeps."""
        for _ in range(self.config.steps_per_frame):
            self.substep()

    def get_particle_positions(self) -> np.ndarray:
        """Return particle positions as numpy array (N, 2)."""
        return self.x.to_numpy()

    def get_particle_volumes(self) -> np.ndarray:
        """Return particle volume ratios (J) as numpy array."""
        return self.J.to_numpy()

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._init_particles()


def init_taichi(arch: str = "auto") -> None:
    """Initialize Taichi with specified architecture.

    Args:
        arch: "auto", "cpu", "cuda", or "vulkan"
    """
    arch_map = {
        "auto": None,
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
    }

    if arch == "auto":
        # Try GPU first, fall back to CPU
        try:
            ti.init(arch=ti.cuda)
            print("Taichi initialized with CUDA backend")
        except Exception:
            try:
                ti.init(arch=ti.vulkan)
                print("Taichi initialized with Vulkan backend")
            except Exception:
                ti.init(arch=ti.cpu)
                print("Taichi initialized with CPU backend")
    else:
        ti.init(arch=arch_map[arch])
        print(f"Taichi initialized with {arch} backend")
