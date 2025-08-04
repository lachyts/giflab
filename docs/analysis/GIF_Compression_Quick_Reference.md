# GIF Compression Quick Reference Guide

## 🚀 Best Overall Settings

### Maximum Compression
```bash
# Animately (Best overall)
animately -i input.gif -o output.gif -l 90 -p 16 -f 0.15

# Alternative: ImageMagick Riemersma
magick input.gif -dither Riemersma -colors 8 output.gif
```

### Balanced Quality/Size
```bash
# Gifski (Best quality balance)
gifski --quality 60 -o output.gif input.gif

# Alternative: Animately
animately -i input.gif -o output.gif -l 60 -p 32
```

## 🎯 Content-Specific Recommendations

### Gradients & Smooth Content
```bash
# ALWAYS use dithering
magick input.gif -dither Riemersma -colors 16 output.gif
```

### Solid Colors & High Contrast
```bash
# NEVER use dithering
gifsicle -O2 --colors 16 --no-dither input.gif -o output.gif
```

### Noisy/Complex Content
```bash
# Use aggressive Bayer dithering
ffmpeg -i input.gif -vf "palettegen=max_colors=16" palette.png
ffmpeg -i input.gif -i palette.png -lavfi "paletteuse=dither=bayer:bayer_scale=4" output.gif
rm palette.png
```

## ✅ Essential Methods to Test
1. **ImageMagick Riemersma** (best all-around)
2. **FFmpeg Sierra2** (quality/size balance)
3. **FFmpeg Floyd-Steinberg** (quality baseline)
4. **No dithering** (size baseline)

## ❌ Skip These Methods
- ImageMagick O2x2, O3x3, O4x4, O8x8 (redundant)
- ImageMagick H4x4a, H6x6a, H8x8a (redundant)
- FFmpeg Bayer Scale 0 (poor quality)
- Gifsicle O3 optimization (minimal benefit, 80% slower)

## 🔧 Tool Settings

### Gifsicle
- **Use O2 optimization** (not O3)
- `gifsicle -O2 --colors 16 --no-dither input.gif -o output.gif`

### FFmpeg Bayer Scales
- **Scale 1-2**: Quality priority
- **Scale 3**: Balanced
- **Scale 4-5**: Maximum compression (noisy content)

### Color Reduction Levels
- **32 colors**: Good balance
- **16 colors**: Aggressive compression
- **8 colors**: Maximum compression (quality loss acceptable)

## 📊 Expected Results (from 4.08MB test file)
- **Animately max**: 2.2MB (46% reduction)
- **ImageMagick**: 2.8MB (31% reduction)
- **Gifski**: 3.0MB (26% reduction)
- **FFmpeg**: 3.4MB (17% reduction)
- **Gifsicle**: 3.7MB (9% reduction)
