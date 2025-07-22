from .imagemagick import color_reduce as imagemagick_color_reduce, frame_reduce as imagemagick_frame_reduce, lossy_compress as imagemagick_lossy_compress
from .ffmpeg import color_reduce as ffmpeg_color_reduce, frame_reduce as ffmpeg_frame_reduce, lossy_compress as ffmpeg_lossy_compress
from .gifski import lossy_compress as gifski_lossy_compress

__all__ = [
    # ImageMagick
    "imagemagick_color_reduce",
    "imagemagick_frame_reduce",
    "imagemagick_lossy_compress",
    # FFmpeg
    "ffmpeg_color_reduce",
    "ffmpeg_frame_reduce",
    "ffmpeg_lossy_compress",
    # gifski
    "gifski_lossy_compress",
] 