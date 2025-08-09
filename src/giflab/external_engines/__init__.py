from .ffmpeg import color_reduce as ffmpeg_color_reduce
from .ffmpeg import export_png_sequence as ffmpeg_export_png_sequence
from .ffmpeg import frame_reduce as ffmpeg_frame_reduce
from .ffmpeg import lossy_compress as ffmpeg_lossy_compress
from .gifski import lossy_compress as gifski_lossy_compress
from .imagemagick import color_reduce as imagemagick_color_reduce
from .imagemagick import export_png_sequence as imagemagick_export_png_sequence
from .imagemagick import frame_reduce as imagemagick_frame_reduce
from .imagemagick import lossy_compress as imagemagick_lossy_compress

__all__ = [
    # ImageMagick
    "imagemagick_color_reduce",
    "imagemagick_frame_reduce",
    "imagemagick_lossy_compress",
    "imagemagick_export_png_sequence",
    # FFmpeg
    "ffmpeg_color_reduce",
    "ffmpeg_frame_reduce",
    "ffmpeg_lossy_compress",
    "ffmpeg_export_png_sequence",
    # gifski
    "gifski_lossy_compress",
]
