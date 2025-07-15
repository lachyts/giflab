#!/bin/bash
# GifLab Setup Verification Script for macOS/Linux
# Run this after installing all dependencies to verify everything works

echo "🎞️ GifLab Setup Verification"
echo "================================"

# Check Python
echo -e "\n1. Checking Python installation..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1)
    if [[ $python_version =~ Python\ 3\.(1[1-9]|[2-9][0-9]) ]]; then
        echo "✅ Python found: $python_version"
    else
        echo "❌ Python 3.11+ required. Found: $python_version"
        echo "   Please install Python 3.11+ from https://www.python.org/downloads/"
    fi
else
    echo "❌ Python not found in PATH"
    echo "   Please install Python 3.11+ and add to PATH"
fi

# Check pip
echo -e "\n2. Checking pip installation..."
if command -v pip3 &> /dev/null; then
    pip_version=$(pip3 --version 2>&1)
    echo "✅ pip found: $pip_version"
else
    echo "❌ pip not found"
fi

# Check Poetry
echo -e "\n3. Checking Poetry installation..."
if command -v poetry &> /dev/null; then
    poetry_version=$(poetry --version 2>&1)
    echo "✅ Poetry found: $poetry_version"
else
    echo "❌ Poetry not found"
    echo "   Install with: curl -sSL https://install.python-poetry.org | python3 -"
fi

# Check FFmpeg
echo -e "\n4. Checking FFmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version 2>&1 | head -1)
    echo "✅ FFmpeg found: $ffmpeg_version"
else
    echo "❌ FFmpeg not found in PATH"
    echo "   Install with: brew install ffmpeg"
fi

# Check Gifsicle
echo -e "\n5. Checking Gifsicle installation..."
if command -v gifsicle &> /dev/null; then
    gifsicle_version=$(gifsicle --version 2>&1)
    echo "✅ Gifsicle found: $gifsicle_version"
else
    echo "❌ Gifsicle not found in PATH"
    echo "   Install with: brew install gifsicle"
fi

# Check directory structure
echo -e "\n6. Checking directory structure..."
required_dirs=("data/raw" "data/renders" "data/csv" "seed" "logs")
all_good=true

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir exists"
    else
        echo "❌ $dir missing"
        all_good=false
    fi
done

if [ "$all_good" = true ]; then
    echo "✅ All required directories exist"
fi

# Check pyproject.toml
echo -e "\n7. Checking project configuration..."
if [ -f "pyproject.toml" ]; then
    echo "✅ pyproject.toml found"
else
    echo "❌ pyproject.toml missing"
fi

# Check Poetry virtual environment
echo -e "\n8. Checking Poetry virtual environment..."
if command -v poetry &> /dev/null; then
    if poetry env info &> /dev/null; then
        venv_info=$(poetry env info --path 2>&1)
        echo "✅ Poetry virtual environment: $venv_info"
    else
        echo "❌ Poetry virtual environment not found"
        echo "   Run: poetry install"
    fi
else
    echo "❌ Poetry not available"
fi

echo -e "\n🎯 Next Steps:"
echo "1. If any checks failed, install the missing dependencies"
echo "2. Run: poetry install"
echo "3. Add some GIF files to data/raw/"
echo "4. Start with: poetry run jupyter notebook"
echo "5. Open notebooks/01_explore_dataset.ipynb"

echo -e "\n📚 For detailed instructions, see BEGINNER_GUIDE.md" 