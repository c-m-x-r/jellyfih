"""Side-by-side comparison video: Aurelia baseline vs Gen 0 vs Gen 15."""

from moviepy import VideoFileClip, TextClip, CompositeVideoClip, clips_array

FONT_SIZE = 32
CAPTION_H = 60
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

clips_info = [
    ("output/view_aurelia_web.mp4", "Aurelia aurita (Baseline)"),
    ("output/sim_evolution_log_gen0.mp4", "CMA-ES Gen 0 (Random Init)"),
    ("output/sim_evolution_log_gen15.mp4", "CMA-ES Gen 15 (Evolved)"),
]

labeled = []
for path, caption in clips_info:
    video = VideoFileClip(path)
    w, h = video.size

    label = (
        TextClip(
            text=caption,
            font_size=FONT_SIZE,
            color="white",
            bg_color="black",
            size=(w, CAPTION_H),
            font=FONT,
            text_align="center",
            vertical_align="center",
        )
        .with_duration(video.duration)
        .with_fps(video.fps)
    )

    composite = CompositeVideoClip(
        [video.with_position((0, CAPTION_H)), label.with_position((0, 0))],
        size=(w, h + CAPTION_H),
    )
    labeled.append(composite)

final = clips_array([labeled])
final.write_videofile(
    "output/comparison.mp4",
    fps=30,
    codec="libx264",
    audio=False,
)

for clip in labeled:
    clip.close()

print("Wrote output/comparison.mp4")
