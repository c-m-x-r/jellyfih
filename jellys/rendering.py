"""Headless rendering for MPM simulation output.

Generates PNG frames and animated GIFs using PIL.
Color-codes particles by material: fluid (blue), body (pink), actuator (bright pink).
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Optional, Tuple

from .config import RenderConfig


class Renderer:
    """Headless particle renderer with multi-material support."""

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.frames: List[Image.Image] = []
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
        """Render one frame from particle positions in [0, 1] normalized coords."""
        cfg = self.config
        w, h = cfg.width, cfg.height

        img = Image.new("RGB", (w, h), cfg.background_color)
        draw = ImageDraw.Draw(img)

        # Water level indicator
        if water_level is not None and cfg.draw_water_level:
            wy = int((1.0 - water_level) * h)
            draw.line([(0, wy), (w, wy)], fill=cfg.water_level_color, width=1)

        # Particle positions -> pixel coords (flip Y: sim bottom=0, image top=0)
        px = (positions[:, 0] * w).astype(int)
        py = ((1.0 - positions[:, 1]) * h).astype(int)

        n = len(positions)
        if materials is None:
            materials = np.zeros(n, dtype=np.int32)
        if actuators is None:
            actuators = np.zeros(n, dtype=np.int32)

        r = cfg.particle_radius
        for i, (x, y) in enumerate(zip(px, py)):
            if 0 <= x < w and 0 <= y < h:
                if materials[i] == 0:
                    color = cfg.particle_color
                elif actuators[i] == 1:
                    color = cfg.actuator_color
                else:
                    color = cfg.jelly_color
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

        if cfg.save_frames:
            img.save(self.output_path / f"frame_{frame_num:04d}.png")

        self.frames.append(img.copy())
        return img

    def save_gif(self, filename: str = "simulation.gif"):
        if not self.frames:
            print("No frames to save")
            return
        gif_path = self.output_path / filename
        ms_per_frame = int(1000 / self.config.fps)
        self.frames[0].save(
            gif_path, save_all=True, append_images=self.frames[1:],
            duration=ms_per_frame, loop=0,
        )
        print(f"Saved GIF: {gif_path} ({len(self.frames)} frames)")

    def clear_frames(self):
        self.frames = []


def render_simulation(
    solver,
    n_frames: int,
    render_config: Optional[RenderConfig] = None,
    water_level: Optional[float] = None,
    verbose: bool = True,
    track_jellyfish: bool = False,
) -> Tuple[Renderer, Optional[List[Tuple[float, float]]]]:
    """Run simulation loop, rendering each frame.

    Returns (renderer, optional jellyfish CoM trajectory).
    """
    renderer = Renderer(render_config)
    trajectory = [] if track_jellyfish else None

    for frame in range(n_frames):
        positions = solver.get_particle_positions()
        materials = solver.get_particle_materials()
        actuators = solver.get_particle_actuators()

        renderer.render_frame(positions, frame,
                              materials=materials, actuators=actuators,
                              water_level=water_level)

        if track_jellyfish:
            trajectory.append(solver.get_jellyfish_center_of_mass())

        solver.step()

        if verbose and (frame + 1) % 10 == 0:
            msg = f"Frame {frame + 1}/{n_frames}"
            if trajectory:
                com = trajectory[-1]
                msg += f" | Jellyfish CoM: ({com[0]:.3f}, {com[1]:.3f})"
            print(msg)

    if render_config is None or render_config.save_gif:
        renderer.save_gif()

    return renderer, trajectory
