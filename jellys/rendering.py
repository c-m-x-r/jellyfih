"""Headless rendering for MPM simulation output.

Generates PNG frames and animated GIFs using PIL.
Supports multi-material visualization (fluid, jellyfish, actuators).
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Optional, Tuple

from .config import RenderConfig


class Renderer:
    """Headless renderer for particle visualization with multi-material support."""

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.frames: List[Image.Image] = []

        # Ensure output directory exists
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def render_frame(
        self,
        positions: np.ndarray,
        frame_num: int,
        materials: Optional[np.ndarray] = None,
        actuators: Optional[np.ndarray] = None,
        water_level: Optional[float] = None,
    ) -> Image.Image:
        """Render a single frame from particle positions.

        Args:
            positions: (N, 2) array of particle positions in [0, 1] normalized coords
            frame_num: Frame number for saving
            materials: (N,) array of material types (0=fluid, 1=jelly)
            actuators: (N,) array of actuator flags (1=actuator)
            water_level: Y-coordinate of water surface for visualization

        Returns:
            PIL Image of the rendered frame
        """
        cfg = self.config
        width, height = cfg.width, cfg.height

        # Create image with background
        img = Image.new("RGB", (width, height), cfg.background_color)
        draw = ImageDraw.Draw(img)

        # Draw water level line if specified
        if water_level is not None and cfg.draw_water_level:
            water_y = int((1.0 - water_level) * height)
            draw.line([(0, water_y), (width, water_y)], fill=cfg.water_level_color, width=1)

        # Convert normalized positions to pixel coordinates
        # Flip Y axis (simulation Y=0 is bottom, image Y=0 is top)
        pixel_x = (positions[:, 0] * width).astype(int)
        pixel_y = ((1.0 - positions[:, 1]) * height).astype(int)

        radius = cfg.particle_radius

        # Determine colors per particle
        n_particles = len(positions)
        if materials is None:
            materials = np.zeros(n_particles, dtype=np.int32)
        if actuators is None:
            actuators = np.zeros(n_particles, dtype=np.int32)

        # Draw particles
        for i, (px, py) in enumerate(zip(pixel_x, pixel_y)):
            if 0 <= px < width and 0 <= py < height:
                # Choose color based on material and actuator status
                if materials[i] == 0:  # Fluid
                    color = cfg.particle_color
                elif actuators[i] == 1:  # Jellyfish actuator (rim)
                    color = cfg.actuator_color
                else:  # Jellyfish body
                    color = cfg.jelly_color

                draw.ellipse(
                    [px - radius, py - radius, px + radius, py + radius],
                    fill=color
                )

        # Save frame if configured
        if cfg.save_frames:
            frame_path = self.output_path / f"frame_{frame_num:04d}.png"
            img.save(frame_path)

        # Store for GIF generation
        self.frames.append(img.copy())

        return img

    def save_gif(self, filename: str = "simulation.gif"):
        """Save accumulated frames as animated GIF.

        Args:
            filename: Output GIF filename
        """
        if not self.frames:
            print("No frames to save!")
            return

        gif_path = self.output_path / filename
        frame_duration = int(1000 / self.config.fps)  # ms per frame

        # Save as GIF
        self.frames[0].save(
            gif_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=frame_duration,
            loop=0  # Loop forever
        )
        print(f"Saved GIF: {gif_path} ({len(self.frames)} frames)")

    def clear_frames(self):
        """Clear stored frames to free memory."""
        self.frames = []

    def get_last_frame(self) -> Optional[Image.Image]:
        """Get the most recent rendered frame."""
        return self.frames[-1] if self.frames else None


def render_simulation(
    solver,
    n_frames: int,
    render_config: Optional[RenderConfig] = None,
    water_level: Optional[float] = None,
    verbose: bool = True,
    track_jellyfish: bool = False,
) -> Tuple[Renderer, Optional[List[Tuple[float, float]]]]:
    """Run simulation and render all frames with multi-material support.

    Args:
        solver: MPMSolver instance
        n_frames: Number of frames to simulate
        render_config: Rendering configuration
        water_level: Y-coordinate of water surface
        verbose: Print progress
        track_jellyfish: If True, record jellyfish center-of-mass over time

    Returns:
        Tuple of (Renderer with frames, optional list of (x, y) CoM positions)
    """
    renderer = Renderer(render_config)
    jellyfish_trajectory = [] if track_jellyfish else None

    # Check if solver has material support
    has_materials = hasattr(solver, 'get_particle_materials')

    for frame in range(n_frames):
        # Get current state
        positions = solver.get_particle_positions()

        # Get materials and actuators if available
        materials = solver.get_particle_materials() if has_materials else None
        actuators = solver.get_particle_actuators() if has_materials else None

        # Render frame
        renderer.render_frame(
            positions, frame,
            materials=materials,
            actuators=actuators,
            water_level=water_level,
        )

        # Track jellyfish center of mass
        if track_jellyfish and hasattr(solver, 'get_jellyfish_center_of_mass'):
            com = solver.get_jellyfish_center_of_mass()
            jellyfish_trajectory.append(com)

        # Advance simulation
        solver.step()

        if verbose and (frame + 1) % 10 == 0:
            if track_jellyfish and jellyfish_trajectory:
                com = jellyfish_trajectory[-1]
                print(f"Frame {frame + 1}/{n_frames} | Jellyfish CoM: ({com[0]:.3f}, {com[1]:.3f})")
            else:
                print(f"Frame {frame + 1}/{n_frames}")

    # Save GIF if configured
    if render_config is None or render_config.save_gif:
        renderer.save_gif()

    return renderer, jellyfish_trajectory
