"""2D MPM Solver using Taichi - DiffTaichi style.

This module implements a Material Point Method solver for fluid and soft body
simulation, following the DiffTaichi (Hu et al., ICLR 2020) patterns.

Supports:
- Weakly compressible fluid
- Fixed Corotated elastic solid
- Neo-Hookean elastic solid
- Active strain actuation for soft robots

Architecture is designed to support future differentiability.
"""

import taichi as ti
import numpy as np
from typing import Optional, Tuple
import math

from .config import SimulationConfig, default_config

# Material type constants
MATERIAL_FLUID = 0
MATERIAL_JELLY = 1

# Constitutive model constants
MODEL_FIXED_COROTATED = 0
MODEL_NEO_HOOKEAN = 1


@ti.data_oriented
class MPMSolver:
    """2D MPM Solver for fluid and soft body simulation.

    Supports multiple material types with different constitutive models.
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
        self.time = 0.0  # Simulation time for actuation

        # Determine total particle count (water + jellyfish)
        self._compute_particle_counts()
        self._init_fields()
        self._init_particles()

    def _compute_particle_counts(self) -> None:
        """Compute particle counts for water and jellyfish."""
        cfg = self.config

        # Water particles
        self.n_water = cfg.n_particles

        # Jellyfish particles (if enabled)
        if cfg.jellyfish.enabled:
            from .jellyfish import compute_jellyfish_particle_count
            self.n_jelly = compute_jellyfish_particle_count(
                cfg.jellyfish.geometry, cfg.grid.dx
            )
        else:
            self.n_jelly = 0

        self.n_particles = self.n_water + self.n_jelly

    def _init_fields(self) -> None:
        """Initialize Taichi fields for particles and grid."""
        cfg = self.config
        n_particles = self.n_particles
        n_grid = cfg.grid.resolution

        # Particle fields
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # position
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # velocity
        self.C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # APIC affine
        self.F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # deformation gradient
        self.J = ti.field(dtype=ti.f32, shape=n_particles)  # volume ratio

        # Material type and actuation
        self.material = ti.field(dtype=ti.i32, shape=n_particles)  # 0=fluid, 1=jelly
        self.is_actuator = ti.field(dtype=ti.i32, shape=n_particles)  # 1=rim actuator

        # Grid fields
        self.grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

        # Store config values as Taichi-friendly constants
        self.n_grid = n_grid
        self.dx = cfg.grid.dx
        self.inv_dx = cfg.grid.inv_dx
        self.dt = cfg.physics.dt
        self.p_mass = 1.0
        self.p_vol = (self.dx ** 2) / cfg.particles.particles_per_cell

        # Fluid parameters
        E_fluid = cfg.physics.E
        nu_fluid = cfg.physics.nu
        self.bulk_modulus = E_fluid / (2.0 * (1.0 - nu_fluid))

        # Jellyfish solid parameters
        if cfg.jellyfish.enabled:
            jelly_mat = cfg.jellyfish.material
            self.jelly_mu = jelly_mat.mu
            self.jelly_lambda = jelly_mat.lambda_
            self.jelly_model = MODEL_FIXED_COROTATED if jelly_mat.model == "fixed_corotated" else MODEL_NEO_HOOKEAN

            # Actuation parameters
            act = cfg.jellyfish.actuation
            self.actuation_freq = act.frequency
            self.actuation_amp = act.amplitude
            self.actuation_phase = act.phase_offset
        else:
            self.jelly_mu = 0.0
            self.jelly_lambda = 0.0
            self.jelly_model = MODEL_FIXED_COROTATED
            self.actuation_freq = 0.0
            self.actuation_amp = 0.0
            self.actuation_phase = 0.0

        # Gravity
        self.gravity = ti.Vector(cfg.physics.gravity)

    def _init_particles(self) -> None:
        """Initialize particle positions for water and jellyfish."""
        cfg = self.config
        rng = np.random.default_rng(self.seed)

        # === Water particles ===
        x_min, y_min, x_max, y_max = cfg.particles.water_region
        water_positions = np.zeros((self.n_water, 2), dtype=np.float32)

        cells_x = int((x_max - x_min) * cfg.grid.resolution)
        cells_y = int((y_max - y_min) * cfg.grid.resolution)
        ppc = cfg.particles.particles_per_cell

        idx = 0
        for i in range(cells_x):
            for j in range(cells_y):
                cell_x = x_min + (i + 0.5) * self.dx
                cell_y = y_min + (j + 0.5) * self.dx
                for _ in range(ppc):
                    if idx < self.n_water:
                        jitter_x = (rng.random() - 0.5) * self.dx * 0.8
                        jitter_y = (rng.random() - 0.5) * self.dx * 0.8
                        water_positions[idx] = [cell_x + jitter_x, cell_y + jitter_y]
                        idx += 1

        # === Jellyfish particles ===
        if cfg.jellyfish.enabled and self.n_jelly > 0:
            from .jellyfish import generate_jellyfish_particles

            jelly_positions, jelly_actuators = generate_jellyfish_particles(
                cfg.jellyfish.geometry,
                self.dx,
                seed=self.seed + 1000,
            )

            # Verify particle count fits in allocated fields
            actual_jelly = len(jelly_positions)
            if actual_jelly > self.n_jelly:
                # Truncate to fit allocated space (with warning)
                print(f"Warning: Generated {actual_jelly} jellyfish particles, truncating to {self.n_jelly}")
                jelly_positions = jelly_positions[:self.n_jelly]
                jelly_actuators = jelly_actuators[:self.n_jelly]
            elif actual_jelly < self.n_jelly:
                # Update count to actual
                self.n_jelly = actual_jelly
                self.n_particles = self.n_water + self.n_jelly

            # Combine positions
            all_positions = np.vstack([water_positions, jelly_positions])

            # Material types
            materials = np.zeros(self.n_particles, dtype=np.int32)
            materials[self.n_water:] = MATERIAL_JELLY

            # Actuator flags
            actuators = np.zeros(self.n_particles, dtype=np.int32)
            actuators[self.n_water:] = jelly_actuators
        else:
            all_positions = water_positions
            materials = np.zeros(self.n_particles, dtype=np.int32)
            actuators = np.zeros(self.n_particles, dtype=np.int32)

        # Copy to Taichi fields
        self.x.from_numpy(all_positions[:self.n_particles])
        self.material.from_numpy(materials[:self.n_particles])
        self.is_actuator.from_numpy(actuators[:self.n_particles])

        # Initialize state
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

    @ti.func
    def _compute_fluid_stress(self, J: ti.f32) -> ti.Matrix:
        """Compute Cauchy stress for weakly compressible fluid."""
        pressure = -self.bulk_modulus * (J - 1.0) / J
        return ti.Matrix([[pressure, 0.0], [0.0, pressure]])

    @ti.func
    def _clamp_F(self, F: ti.Matrix) -> ti.Matrix:
        """Clamp deformation gradient using SVD to prevent inversion.

        Ensures singular values stay in safe range [0.2, 5.0].
        """
        U, sig, V = ti.svd(F)
        # Clamp singular values
        sig_clamped = ti.Vector([
            ti.max(ti.min(sig[0], 5.0), 0.2),
            ti.max(ti.min(sig[1], 5.0), 0.2)
        ])
        # Reconstruct clamped F
        return U @ ti.Matrix([[sig_clamped[0], 0.0], [0.0, sig_clamped[1]]]) @ V.transpose()

    @ti.func
    def _compute_fixed_corotated_stress(self, F: ti.Matrix) -> ti.Matrix:
        """Compute Cauchy stress using Fixed Corotated model.

        Handles large rotations and prevents element inversion.
        Following DiffTaichi formulation.
        """
        # Clamp F to prevent numerical issues
        F_safe = self._clamp_F(F)

        # SVD decomposition: F = U @ S @ V^T
        U, sig, V = ti.svd(F_safe)
        R = U @ V.transpose()  # Rotation component

        J = sig[0] * sig[1]  # Determinant from singular values
        J_safe = ti.max(J, 0.2)

        # First Piola-Kirchhoff stress: P = 2*mu*(F-R) + lambda*(J-1)*J*F^-T
        # For stability, compute F^-T via SVD: F^-T = U @ diag(1/sig) @ V^T transposed
        sig_inv = ti.Vector([1.0 / sig[0], 1.0 / sig[1]])
        F_inv_T = V @ ti.Matrix([[sig_inv[0], 0.0], [0.0, sig_inv[1]]]) @ U.transpose()

        P = 2.0 * self.jelly_mu * (F_safe - R) + self.jelly_lambda * (J_safe - 1.0) * F_inv_T

        # Convert to Cauchy stress: sigma = (1/J) * P @ F^T
        stress = (1.0 / J_safe) * P @ F_safe.transpose()

        return stress

    @ti.func
    def _compute_neo_hookean_stress(self, F: ti.Matrix) -> ti.Matrix:
        """Compute Cauchy stress using Neo-Hookean model.

        Uses SVD-based clamping for numerical stability.
        """
        # Clamp F to prevent numerical issues
        F_safe = self._clamp_F(F)

        # SVD for safe inverse computation
        U, sig, V = ti.svd(F_safe)
        J = sig[0] * sig[1]
        J_safe = ti.max(J, 0.2)

        # F^-T via SVD
        sig_inv = ti.Vector([1.0 / sig[0], 1.0 / sig[1]])
        F_inv_T = V @ ti.Matrix([[sig_inv[0], 0.0], [0.0, sig_inv[1]]]) @ U.transpose()

        # Neo-Hookean: P = mu*(F - F^-T) + lambda*ln(J)*F^-T
        P = self.jelly_mu * (F_safe - F_inv_T) + self.jelly_lambda * ti.log(J_safe) * F_inv_T

        # Convert to Cauchy: sigma = (1/J) * P * F^T
        stress = (1.0 / J_safe) * P @ F_safe.transpose()

        return stress

    @ti.func
    def _get_actuation_contraction(self, time: ti.f32) -> ti.f32:
        """Compute current contraction factor from sinusoidal pattern.

        Returns value in [0, amplitude] - only contracts, doesn't expand.
        Amplitude is clamped to safe range.
        """
        # Clamp amplitude to safe range to prevent numerical issues
        safe_amp = ti.min(self.actuation_amp, 0.5)

        phase = 2.0 * math.pi * self.actuation_freq * time + self.actuation_phase
        # Use -sin to start with contraction, then max with 0 for contraction-only
        raw = -ti.sin(phase)
        return safe_amp * ti.max(raw, 0.0)

    @ti.func
    def _apply_active_strain(self, F: ti.Matrix, contraction: ti.f32) -> ti.Matrix:
        """Apply active contractile strain to deformation gradient.

        Simulates muscle contraction by modifying the rest configuration.
        For bell contraction: the rest shape wants to be smaller horizontally.
        F_elastic = F @ F_active, where F_active = diag(1-c, 1) represents
        the desired contraction of the rest state.
        """
        # Active strain: multiply F by contraction factor
        # When contraction > 0, the material "wants" to be horizontally contracted
        # This creates stress that pulls the bell inward
        scale = 1.0 - contraction  # 0.3 contraction -> 0.7 scale

        F_active = ti.Matrix([
            [F[0, 0] * scale, F[0, 1] * scale],
            [F[1, 0], F[1, 1]]
        ])
        return F_active

    @ti.kernel
    def _p2g(self, time: ti.f32):
        """Particle-to-Grid transfer with multi-material support."""
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)

            # Quadratic B-spline weights
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

            # Compute stress based on material type
            stress = ti.Matrix.zero(ti.f32, 2, 2)

            if self.material[p] == MATERIAL_FLUID:
                stress = self._compute_fluid_stress(self.J[p])
            else:
                # Jellyfish solid - apply actuation if rim particle
                F = self.F[p]
                if self.is_actuator[p] == 1:
                    contraction = self._get_actuation_contraction(time)
                    F = self._apply_active_strain(F, contraction)

                # Choose constitutive model
                if self.jelly_model == MODEL_FIXED_COROTATED:
                    stress = self._compute_fixed_corotated_stress(F)
                else:
                    stress = self._compute_neo_hookean_stress(F)

            # APIC momentum transfer
            affine = stress * self.p_vol * self.inv_dx + self.p_mass * self.C[p]

            # Scatter to grid
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = wx[i] * wy[j]

                grid_idx = base + offset
                if 0 <= grid_idx.x < self.n_grid and 0 <= grid_idx.y < self.n_grid:
                    self.grid_v[grid_idx] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                    self.grid_m[grid_idx] += weight * self.p_mass

    @ti.kernel
    def _grid_op(self):
        """Grid operations: normalize, gravity, boundaries."""
        for i, j in self.grid_v:
            if self.grid_m[i, j] > 1e-10:
                self.grid_v[i, j] /= self.grid_m[i, j]
                self.grid_v[i, j] += self.dt * self.gravity

                bound = 3
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
        """Grid-to-Particle transfer with material-aware F update."""
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)

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

            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = ti.cast(offset, ti.f32) - fx
                weight = wx[i] * wy[j]

                grid_idx = base + offset
                if 0 <= grid_idx.x < self.n_grid and 0 <= grid_idx.y < self.n_grid:
                    g_v = self.grid_v[grid_idx]
                    new_v += weight * g_v
                    new_C += 4.0 * self.inv_dx * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v

            # Update deformation gradient
            F_new = (ti.Matrix.identity(ti.f32, 2) + self.dt * new_C) @ self.F[p]

            if self.material[p] == MATERIAL_FLUID:
                # For fluid: track volume only, reset F to isotropic
                J_new = F_new.determinant()
                J_clamped = ti.max(ti.min(J_new, 4.0), 0.1)
                sqrt_J = ti.sqrt(J_clamped)
                self.F[p] = ti.Matrix([[sqrt_J, 0.0], [0.0, sqrt_J]])
                self.J[p] = J_clamped
            else:
                # For solid: clamp F using SVD to keep J in safe range
                # This maintains consistency between F and J
                U, sig, V = ti.svd(F_new)
                sig_clamped = ti.Vector([
                    ti.max(ti.min(sig[0], 2.0), 0.5),
                    ti.max(ti.min(sig[1], 2.0), 0.5)
                ])
                self.F[p] = U @ ti.Matrix([[sig_clamped[0], 0.0], [0.0, sig_clamped[1]]]) @ V.transpose()
                self.J[p] = sig_clamped[0] * sig_clamped[1]

    def substep(self) -> None:
        """Execute one simulation substep."""
        self._clear_grid()
        self._p2g(self.time)
        self._grid_op()
        self._g2p()
        self.time += self.dt

    def step(self) -> None:
        """Execute one frame's worth of substeps."""
        for _ in range(self.config.steps_per_frame):
            self.substep()

    def get_particle_positions(self) -> np.ndarray:
        """Return particle positions as numpy array (N, 2)."""
        return self.x.to_numpy()[:self.n_particles]

    def get_particle_materials(self) -> np.ndarray:
        """Return particle material types as numpy array."""
        return self.material.to_numpy()[:self.n_particles]

    def get_particle_actuators(self) -> np.ndarray:
        """Return particle actuator flags as numpy array."""
        return self.is_actuator.to_numpy()[:self.n_particles]

    def get_particle_volumes(self) -> np.ndarray:
        """Return particle volume ratios (J) as numpy array."""
        return self.J.to_numpy()[:self.n_particles]

    def get_jellyfish_center_of_mass(self) -> Tuple[float, float]:
        """Compute center of mass of jellyfish particles."""
        if self.n_jelly == 0:
            return (0.0, 0.0)

        positions = self.x.to_numpy()
        jelly_pos = positions[self.n_water:self.n_water + self.n_jelly]
        return (float(jelly_pos[:, 0].mean()), float(jelly_pos[:, 1].mean()))

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.time = 0.0
        self._init_particles()


def init_taichi(arch: str = "auto") -> None:
    """Initialize Taichi with specified architecture."""
    if arch == "auto":
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
        arch_map = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
        ti.init(arch=arch_map[arch])
        print(f"Taichi initialized with {arch} backend")
