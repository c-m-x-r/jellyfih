"""2D MPM (Material Point Method) solver with APIC transfer.

Implements a unified fluid + soft-body simulation following the MLS-MPM
formulation (Hu et al. 2018). Supports:
  - Weakly compressible fluid
  - Fixed Corotated / Neo-Hookean elastic solid
  - Active strain actuation for jellyfish bell contraction

Reference conventions (MLS-MPM):
  - All grid-space quantities use dimensionless coordinates: Xp * inv_dx
  - dpos in P2G and G2P is in grid-space (offset - fx), NOT physical space
  - Stress contribution: -dt * 4 * inv_dx^2 * p_vol * sigma
  - APIC C recovery:     4 * inv_dx * weight * g_v outer dpos
"""

import taichi as ti
import numpy as np
from typing import Optional, Tuple
import math

from .config import SimulationConfig, default_config

# Material types
MATERIAL_FLUID = 0
MATERIAL_JELLY = 1

# Constitutive models
MODEL_FIXED_COROTATED = 0
MODEL_NEO_HOOKEAN = 1


@ti.data_oriented
class MPMSolver:
    """2D MPM solver for coupled fluid-structure jellyfish simulation."""

    def __init__(self, config: Optional[SimulationConfig] = None, seed: int = 42):
        self.config = config or default_config()
        self.seed = seed
        self.time = 0.0

        self._compute_particle_counts()
        self._init_fields()
        self._init_particles()

    # =========================================================================
    # Initialization
    # =========================================================================

    def _compute_particle_counts(self) -> None:
        cfg = self.config
        self.n_water = cfg.n_particles

        if cfg.jellyfish.enabled:
            from .jellyfish import compute_jellyfish_particle_count
            self.n_jelly = compute_jellyfish_particle_count(
                cfg.jellyfish.geometry, cfg.grid.dx
            )
        else:
            self.n_jelly = 0

        self.n_particles = self.n_water + self.n_jelly

    def _init_fields(self) -> None:
        cfg = self.config
        n = self.n_particles
        g = cfg.grid.resolution

        # Per-particle state
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=n)         # position
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=n)         # velocity
        self.C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n)      # APIC affine matrix
        self.F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n)      # deformation gradient
        self.J = ti.field(dtype=ti.f32, shape=n)                    # volume ratio det(F)
        self.material = ti.field(dtype=ti.i32, shape=n)             # 0=fluid, 1=jelly
        self.is_actuator = ti.field(dtype=ti.i32, shape=n)          # 1=rim actuator
        self.p_mass = ti.field(dtype=ti.f32, shape=n)               # per-particle mass

        # Grid state
        self.grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(g, g))
        self.grid_m = ti.field(dtype=ti.f32, shape=(g, g))

        # Cache frequently used scalars
        self.n_grid = g
        self.dx = cfg.grid.dx
        self.inv_dx = cfg.grid.inv_dx
        self.dt = cfg.physics.dt
        self.p_vol = (self.dx ** 2) / cfg.particles.particles_per_cell

        # Fluid bulk modulus: K = E / (3*(1 - 2*nu))
        self.bulk_modulus = cfg.physics.E / (3.0 * (1.0 - 2.0 * cfg.physics.nu))

        # Jellyfish material parameters
        if cfg.jellyfish.enabled:
            mat = cfg.jellyfish.material
            self.jelly_mu = mat.mu
            self.jelly_lambda = mat.lambda_
            self.jelly_density = mat.density
            self.jelly_model = (MODEL_FIXED_COROTATED
                                if mat.model == "fixed_corotated"
                                else MODEL_NEO_HOOKEAN)

            act = cfg.jellyfish.actuation
            self.actuation_freq = act.frequency
            self.actuation_amp = act.amplitude
            self.actuation_phase = act.phase_offset
        else:
            self.jelly_mu = 0.0
            self.jelly_lambda = 0.0
            self.jelly_density = 1.0
            self.jelly_model = MODEL_FIXED_COROTATED
            self.actuation_freq = 0.0
            self.actuation_amp = 0.0
            self.actuation_phase = 0.0

        self.gravity = ti.Vector(cfg.physics.gravity)

    def _init_particles(self) -> None:
        """Spawn water and jellyfish particles. Water is excluded from jellyfish region."""
        cfg = self.config
        rng = np.random.default_rng(self.seed)

        # --- Water particles (with exclusion zone for jellyfish) ---
        x_min, y_min, x_max, y_max = cfg.particles.water_region
        ppc = cfg.particles.particles_per_cell
        cells_x = int((x_max - x_min) * cfg.grid.resolution)
        cells_y = int((y_max - y_min) * cfg.grid.resolution)

        # Precompute jellyfish exclusion zone
        if cfg.jellyfish.enabled:
            from .jellyfish import is_point_in_dome
            jg = cfg.jellyfish.geometry
            jcx, jcy = jg.center
            ja, jb = jg.semi_major, jg.semi_minor
            exclude_jelly = True
        else:
            exclude_jelly = False

        water_positions = []
        for i in range(cells_x):
            for j in range(cells_y):
                cell_x = x_min + (i + 0.5) * self.dx
                cell_y = y_min + (j + 0.5) * self.dx
                for _ in range(ppc):
                    px = cell_x + (rng.random() - 0.5) * self.dx * 0.8
                    py = cell_y + (rng.random() - 0.5) * self.dx * 0.8
                    # Skip particles inside jellyfish dome
                    if exclude_jelly and is_point_in_dome(px, py, jcx, jcy, ja, jb):
                        continue
                    water_positions.append([px, py])

        water_positions = np.array(water_positions, dtype=np.float32)
        self.n_water = len(water_positions)

        # --- Jellyfish particles ---
        if cfg.jellyfish.enabled and self.n_jelly > 0:
            from .jellyfish import generate_jellyfish_particles
            jelly_positions, jelly_actuators = generate_jellyfish_particles(
                cfg.jellyfish.geometry, self.dx, seed=self.seed + 1000,
            )
            actual_jelly = len(jelly_positions)
            if actual_jelly > self.n_jelly:
                print(f"Warning: truncating {actual_jelly} jellyfish particles to {self.n_jelly}")
                jelly_positions = jelly_positions[:self.n_jelly]
                jelly_actuators = jelly_actuators[:self.n_jelly]
            else:
                self.n_jelly = actual_jelly

            self.n_particles = self.n_water + self.n_jelly
            all_positions = np.vstack([water_positions, jelly_positions])

            materials = np.zeros(self.n_particles, dtype=np.int32)
            materials[self.n_water:] = MATERIAL_JELLY

            actuators = np.zeros(self.n_particles, dtype=np.int32)
            actuators[self.n_water:] = jelly_actuators

            # Per-particle mass: water=1.0, jelly=density
            masses = np.ones(self.n_particles, dtype=np.float32)
            masses[self.n_water:] = self.jelly_density
        else:
            self.n_particles = self.n_water
            all_positions = water_positions
            materials = np.zeros(self.n_particles, dtype=np.int32)
            actuators = np.zeros(self.n_particles, dtype=np.int32)
            masses = np.ones(self.n_particles, dtype=np.float32)

        # Copy to Taichi fields
        self.x.from_numpy(all_positions[:self.n_particles])
        self.material.from_numpy(materials[:self.n_particles])
        self.is_actuator.from_numpy(actuators[:self.n_particles])
        self.p_mass.from_numpy(masses[:self.n_particles])
        self._reset_particle_state()

    @ti.kernel
    def _reset_particle_state(self):
        for p in range(self.n_particles):
            self.v[p] = ti.Vector([0.0, 0.0])
            self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            self.F[p] = ti.Matrix.identity(ti.f32, 2)
            self.J[p] = 1.0

    # =========================================================================
    # Constitutive models
    # =========================================================================

    @ti.func
    def _compute_fluid_stress(self, J: ti.f32) -> ti.Matrix:
        """Weakly compressible fluid: isotropic pressure from equation of state."""
        pressure = -self.bulk_modulus * (J - 1.0) / J
        return ti.Matrix([[pressure, 0.0], [0.0, pressure]])

    @ti.func
    def _clamp_F(self, F: ti.Matrix) -> ti.Matrix:
        """SVD-based clamping to prevent inversion. Singular values in [0.3, 3.0]."""
        U, sig, V = ti.svd(F)
        sig_clamped = ti.Vector([
            ti.max(ti.min(sig[0], 3.0), 0.3),
            ti.max(ti.min(sig[1], 3.0), 0.3),
        ])
        return U @ ti.Matrix([[sig_clamped[0], 0.0], [0.0, sig_clamped[1]]]) @ V.transpose()

    @ti.func
    def _compute_fixed_corotated_stress(self, F: ti.Matrix) -> ti.Matrix:
        """Fixed Corotated model (Stomakhin et al. 2013).

        P = 2*mu*(F - R) + lambda*(J - 1)*J*F^{-T}
        sigma = (1/J) * P @ F^T
        """
        F_safe = self._clamp_F(F)
        U, sig, V = ti.svd(F_safe)
        R = U @ V.transpose()

        J = sig[0] * sig[1]
        J_safe = ti.max(J, 0.3)

        # F^{-T} via SVD
        sig_inv = ti.Vector([1.0 / sig[0], 1.0 / sig[1]])
        F_inv_T = V @ ti.Matrix([[sig_inv[0], 0.0], [0.0, sig_inv[1]]]) @ U.transpose()

        # First Piola-Kirchhoff stress (note the J * F_inv_T in volumetric term)
        P = (2.0 * self.jelly_mu * (F_safe - R)
             + self.jelly_lambda * (J_safe - 1.0) * J_safe * F_inv_T)

        # Convert to Cauchy stress
        return (1.0 / J_safe) * P @ F_safe.transpose()

    @ti.func
    def _compute_neo_hookean_stress(self, F: ti.Matrix) -> ti.Matrix:
        """Neo-Hookean hyperelastic model.

        P = mu*(F - F^{-T}) + lambda*ln(J)*F^{-T}
        sigma = (1/J) * P @ F^T
        """
        F_safe = self._clamp_F(F)
        U, sig, V = ti.svd(F_safe)
        J = sig[0] * sig[1]
        J_safe = ti.max(J, 0.3)

        sig_inv = ti.Vector([1.0 / sig[0], 1.0 / sig[1]])
        F_inv_T = V @ ti.Matrix([[sig_inv[0], 0.0], [0.0, sig_inv[1]]]) @ U.transpose()

        P = self.jelly_mu * (F_safe - F_inv_T) + self.jelly_lambda * ti.log(J_safe) * F_inv_T
        return (1.0 / J_safe) * P @ F_safe.transpose()

    # =========================================================================
    # Actuation
    # =========================================================================

    @ti.func
    def _get_actuation_contraction(self, time: ti.f32) -> ti.f32:
        """Sinusoidal contraction pattern. Returns value in [0, amplitude].

        Uses -sin so contraction starts immediately. max(., 0) ensures
        contraction-only (muscles don't push; expansion is passive elastic recoil).
        """
        safe_amp = ti.min(self.actuation_amp, 0.5)
        phase = 2.0 * math.pi * self.actuation_freq * time + self.actuation_phase
        return safe_amp * ti.max(-ti.sin(phase), 0.0)

    @ti.func
    def _apply_active_strain(self, F: ti.Matrix, contraction: ti.f32) -> ti.Matrix:
        """Apply active contractile strain via multiplicative decomposition.

        F = F_elastic @ F_active, where F_active = diag(1-c, 1).
        We want the elastic part: F_elastic = F @ inv(F_active) = F @ diag(1/(1-c), 1).
        The constitutive model then acts on F_elastic, producing stress that
        pulls the bell inward (contraction).
        """
        inv_scale = 1.0 / (1.0 - contraction)
        return ti.Matrix([
            [F[0, 0] * inv_scale, F[0, 1]],
            [F[1, 0] * inv_scale, F[1, 1]],
        ])

    # =========================================================================
    # MPM time integration: P2G -> Grid Ops -> G2P
    # =========================================================================

    @ti.kernel
    def _clear_grid(self):
        for i, j in self.grid_v:
            self.grid_v[i, j] = ti.Vector([0.0, 0.0])
            self.grid_m[i, j] = 0.0

    @ti.kernel
    def _p2g(self, time: ti.f32):
        """Particle-to-Grid: scatter mass, momentum, and stress forces.

        MLS-MPM formulation (Hu et al. 2018):
          affine = -dt * 4 * inv_dx^2 * p_vol * sigma  +  p_mass * C
          grid_v += weight * (p_mass * v_p + affine @ dpos)
          grid_m += weight * p_mass

        dpos is in grid-space: (offset - fx), dimensionless.
        """
        for p in range(self.n_particles):
            # Map particle position to grid-space
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)

            # Quadratic B-spline weights
            wx = ti.Vector([0.5 * (1.5 - fx.x) ** 2,
                            0.75 - (fx.x - 1.0) ** 2,
                            0.5 * (fx.x - 0.5) ** 2])
            wy = ti.Vector([0.5 * (1.5 - fx.y) ** 2,
                            0.75 - (fx.y - 1.0) ** 2,
                            0.5 * (fx.y - 0.5) ** 2])

            # Compute Cauchy stress based on material
            stress = ti.Matrix.zero(ti.f32, 2, 2)
            if self.material[p] == MATERIAL_FLUID:
                stress = self._compute_fluid_stress(self.J[p])
            else:
                F = self.F[p]
                if self.is_actuator[p] == 1:
                    contraction = self._get_actuation_contraction(time)
                    F = self._apply_active_strain(F, contraction)
                if self.jelly_model == MODEL_FIXED_COROTATED:
                    stress = self._compute_fixed_corotated_stress(F)
                else:
                    stress = self._compute_neo_hookean_stress(F)

            # MLS-MPM affine matrix (grid-space dpos convention)
            #   stress term:  -dt * 4 * inv_dx^2 * p_vol * sigma
            #   APIC term:    p_mass * C
            mass_p = self.p_mass[p]
            stress_contrib = -self.dt * 4.0 * self.inv_dx * self.inv_dx * self.p_vol * stress
            affine = stress_contrib + mass_p * self.C[p]

            # Scatter to 3x3 grid neighborhood
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = ti.cast(offset, ti.f32) - fx  # grid-space
                weight = wx[i] * wy[j]
                grid_idx = base + offset

                if 0 <= grid_idx.x < self.n_grid and 0 <= grid_idx.y < self.n_grid:
                    self.grid_v[grid_idx] += weight * (mass_p * self.v[p] + affine @ dpos)
                    self.grid_m[grid_idx] += weight * mass_p

    @ti.kernel
    def _grid_op(self):
        """Grid operations: normalize momentum -> velocity, apply gravity, enforce BCs."""
        for i, j in self.grid_v:
            if self.grid_m[i, j] > 1e-10:
                # Momentum -> velocity
                self.grid_v[i, j] /= self.grid_m[i, j]
                # Gravity
                self.grid_v[i, j] += self.dt * self.gravity

                # Boundary conditions: free-slip walls (3-cell padding)
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
        """Grid-to-Particle: gather velocity, update C and F.

        APIC C matrix: C = 4 * inv_dx * sum(weight * g_v outer dpos)
        where dpos is in grid-space (offset - fx).

        Note: only ONE factor of inv_dx here (not inv_dx^2). The C matrix
        has units of 1/time, matching the deformation gradient update:
        F_new = (I + dt * C) @ F.
        """
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - ti.cast(base, ti.f32)

            wx = ti.Vector([0.5 * (1.5 - fx.x) ** 2,
                            0.75 - (fx.x - 1.0) ** 2,
                            0.5 * (fx.x - 0.5) ** 2])
            wy = ti.Vector([0.5 * (1.5 - fx.y) ** 2,
                            0.75 - (fx.y - 1.0) ** 2,
                            0.5 * (fx.y - 0.5) ** 2])

            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix.zero(ti.f32, 2, 2)

            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = ti.cast(offset, ti.f32) - fx  # grid-space
                weight = wx[i] * wy[j]
                grid_idx = base + offset

                if 0 <= grid_idx.x < self.n_grid and 0 <= grid_idx.y < self.n_grid:
                    g_v = self.grid_v[grid_idx]
                    new_v += weight * g_v
                    # APIC affine recovery: 4 * inv_dx (one power, not two)
                    new_C += 4.0 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v

            # Update deformation gradient: F_new = (I + dt*C) @ F
            F_new = (ti.Matrix.identity(ti.f32, 2) + self.dt * new_C) @ self.F[p]

            if self.material[p] == MATERIAL_FLUID:
                # Fluid: track volume ratio J only, reset F to isotropic
                J_new = F_new.determinant()
                J_clamped = ti.max(ti.min(J_new, 4.0), 0.1)
                sqrt_J = ti.sqrt(J_clamped)
                self.F[p] = ti.Matrix([[sqrt_J, 0.0], [0.0, sqrt_J]])
                self.J[p] = J_clamped
            else:
                # Solid: clamp F via SVD to maintain stability [0.3, 3.0]
                # (consistent with stress clamp range in _clamp_F)
                U, sig, V = ti.svd(F_new)
                sig_clamped = ti.Vector([
                    ti.max(ti.min(sig[0], 3.0), 0.3),
                    ti.max(ti.min(sig[1], 3.0), 0.3),
                ])
                self.F[p] = U @ ti.Matrix([[sig_clamped[0], 0.0],
                                            [0.0, sig_clamped[1]]]) @ V.transpose()
                self.J[p] = sig_clamped[0] * sig_clamped[1]

    # =========================================================================
    # Time stepping
    # =========================================================================

    def substep(self) -> None:
        self._clear_grid()
        self._p2g(self.time)
        self._grid_op()
        self._g2p()
        self.time += self.dt

    def step(self) -> None:
        """Advance one frame (multiple substeps)."""
        for _ in range(self.config.steps_per_frame):
            self.substep()

    # =========================================================================
    # Data access
    # =========================================================================

    def get_particle_positions(self) -> np.ndarray:
        return self.x.to_numpy()[:self.n_particles]

    def get_particle_materials(self) -> np.ndarray:
        return self.material.to_numpy()[:self.n_particles]

    def get_particle_actuators(self) -> np.ndarray:
        return self.is_actuator.to_numpy()[:self.n_particles]

    def get_particle_volumes(self) -> np.ndarray:
        return self.J.to_numpy()[:self.n_particles]

    def get_jellyfish_center_of_mass(self) -> Tuple[float, float]:
        if self.n_jelly == 0:
            return (0.0, 0.0)
        positions = self.x.to_numpy()
        jelly_pos = positions[self.n_water:self.n_water + self.n_jelly]
        return (float(jelly_pos[:, 0].mean()), float(jelly_pos[:, 1].mean()))

    def reset(self) -> None:
        self.time = 0.0
        self._init_particles()


# =============================================================================
# Taichi backend initialization
# =============================================================================


def init_taichi(arch: str = "auto") -> None:
    if arch == "auto":
        try:
            ti.init(arch=ti.cuda)
            print("Taichi: CUDA backend")
        except Exception:
            try:
                ti.init(arch=ti.vulkan)
                print("Taichi: Vulkan backend")
            except Exception:
                ti.init(arch=ti.cpu)
                print("Taichi: CPU backend")
    else:
        arch_map = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
        ti.init(arch=arch_map[arch])
        print(f"Taichi: {arch} backend")
