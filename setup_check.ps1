# GifLab Setup Verification Script
# Run this after installing all dependencies to verify everything works

Write-Host "🎞️ GifLab Setup Verification" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Check Python
Write-Host "`n1. Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.(1[1-9]|[2-9][0-9])") {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Python 3.11+ required. Found: $pythonVersion" -ForegroundColor Red
        Write-Host "   Please install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    Write-Host "   Please install Python 3.11+ and add to PATH" -ForegroundColor Red
}

# Check pip
Write-Host "`n2. Checking pip installation..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ pip not found" -ForegroundColor Red
}

# Check Poetry
Write-Host "`n3. Checking Poetry installation..." -ForegroundColor Yellow
try {
    $poetryVersion = poetry --version 2>&1
    Write-Host "✅ Poetry found: $poetryVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Poetry not found" -ForegroundColor Red
    Write-Host "   Install with: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -" -ForegroundColor Red
}

# Check FFmpeg
Write-Host "`n4. Checking FFmpeg installation..." -ForegroundColor Yellow
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    if ($ffmpegVersion -match "ffmpeg version") {
        Write-Host "✅ FFmpeg found: $ffmpegVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ FFmpeg not found" -ForegroundColor Red
        Write-Host "   Install with: choco install ffmpeg" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ FFmpeg not found in PATH" -ForegroundColor Red
    Write-Host "   Install with: choco install ffmpeg" -ForegroundColor Red
}

# Check Gifsicle
Write-Host "`n5. Checking Gifsicle installation..." -ForegroundColor Yellow
try {
    $gifsicleVersion = gifsicle --version 2>&1
    if ($gifsicleVersion -match "Gifsicle") {
        Write-Host "✅ Gifsicle found: $gifsicleVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Gifsicle not found" -ForegroundColor Red
        Write-Host "   Install with: choco install gifsicle" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Gifsicle not found in PATH" -ForegroundColor Red
    Write-Host "   Install with: choco install gifsicle" -ForegroundColor Red
}

# Check directory structure
Write-Host "`n6. Checking directory structure..." -ForegroundColor Yellow
$requiredDirs = @("data/raw", "data/renders", "data/csv", "seed", "logs")
$allGood = $true

foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-Host "✅ $dir exists" -ForegroundColor Green
    } else {
        Write-Host "❌ $dir missing" -ForegroundColor Red
        $allGood = $false
    }
}

if ($allGood) {
    Write-Host "✅ All required directories exist" -ForegroundColor Green
}

# Check pyproject.toml
Write-Host "`n7. Checking project configuration..." -ForegroundColor Yellow
if (Test-Path "pyproject.toml") {
    Write-Host "✅ pyproject.toml found" -ForegroundColor Green
} else {
    Write-Host "❌ pyproject.toml missing" -ForegroundColor Red
}

Write-Host "`n🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "1. If any checks failed, install the missing dependencies" -ForegroundColor White
Write-Host "2. Run: poetry install" -ForegroundColor White
Write-Host "3. Add some GIF files to data/raw/" -ForegroundColor White
Write-Host "4. Start with: jupyter notebook" -ForegroundColor White
Write-Host "5. Open notebooks/01_explore_dataset.ipynb" -ForegroundColor White

Write-Host "`n📚 For detailed instructions, see BEGINNER_GUIDE.md" -ForegroundColor Cyan