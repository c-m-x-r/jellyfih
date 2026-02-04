"""Headless rendering for MPM simulation output.

Generates PNG frames and animated GIFs using PIL.
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Optional

from .config import RenderConfig


class Renderer:
    """Headless renderer for particle visualization."""

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.frames: List[Image.Image] = []

        # Ensure output directory exists
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def render_frame(self, positions: np.ndarray, frame_num: int) -> Image.Image:
        """Render a single frame from particle positions.

        Args:
            positions: (N, 2) array of particle positions in [0, 1] normalized coords
            frame_num: Frame number for saving

        Returns:
            PIL Image of the rendered frame
        """
        cfg = self.config
        width, height = cfg.width, cfg.height

        # Create image with background
        img = Image.new("RGB", (width, height), cfg.background_color)
        draw = ImageDraw.Draw(img)

        # Convert normalized positions to pixel coordinates
        # Flip Y axis (simulation Y=0 is bottom, image Y=0 is top)
        pixel_x = (positions[:, 0] * width).astype(int)
        pixel_y = ((1.0 - positions[:, 1]) * height).astype(int)

        # Draw particles as circles
        radius = cfg.particle_radius
        color = cfg.particle_color

        for px, py in zip(pixel_x, pixel_y):
            # Clip to image bounds
            if 0 <= px < width and 0 <= py < height:
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
    verbose: bool = True
) -> Renderer:
    """Convenience function to run simulation and render all frames.

    Args:
        solver: MPMSolver instance
        n_frames: Number of frames to simulate
        render_config: Rendering configuration
        verbose: Print progress

    Returns:
        Renderer with all frames captured
    """
    renderer = Renderer(render_config)

    for frame in range(n_frames):
        # Get current state and render
        positions = solver.get_particle_positions()
        renderer.render_frame(positions, frame)

        # Advance simulation
        solver.step()

        if verbose and (frame + 1) % 10 == 0:
            print(f"Frame {frame + 1}/{n_frames}")

    # Save GIF if configured
    if render_config is None or render_config.save_gif:
        renderer.save_gif()

    return renderer
