# 🎞️ GifLab Setup Guide for Windows

**Complete step-by-step installation guide for Windows users**

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- Windows 10/11 (version 1709 or later for winget)
- Internet connection
- At least 2GB free disk space

## 🚀 Step-by-Step Installation (Modern Best Practice)

### Step 1: Install Python 3.11+ (Recommended Method)

**Using winget (Modern Windows Best Practice):**
```powershell
# Install Python 3.11 (recommended for stability)
winget install Python.Python.3.11

# Or install Python 3.12 (latest)
winget install Python.Python.3.12
```

**Alternative Methods:**
- **Microsoft Store**: Search "Python 3.11" and install
- **Official Installer**: https://www.python.org/downloads/ (if winget fails)

**Verify Installation:**
```powershell
python --version
# Should show: Python 3.11.x or higher
```

### Step 2: Install System Dependencies

**Using winget (Recommended):**
```powershell
# Install FFmpeg (video processing)
winget install Gyan.FFmpeg

# Install Git (if not already installed)
winget install Git.Git
```

**Note**: Gifsicle is not available via winget, so we'll install it via Chocolatey or manually.

### Step 3: Install Gifsicle

**Option A: Using Chocolatey (if you prefer package managers)**
```powershell
# Install Chocolatey first (run as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Then install Gifsicle
choco install gifsicle
```

**Option B: Manual Installation**
1. Download from: https://eternallybored.org/misc/gifsicle/
2. Extract to a folder (e.g., `C:\Program Files\gifsicle\`)
3. Add to PATH: `C:\Program Files\gifsicle\`

### Step 4: Install Poetry

**Using official installer (Poetry is not available in winget):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

**Alternative method using pip:**
```powershell
pip install poetry
```

**Verify Poetry:**
```powershell
poetry --version
# Should show: Poetry (version x.x.x)
```

### Step 5: Install GifLab Dependencies

1. Navigate to your GifLab directory:
```powershell
cd C:\Users\lachl\repos\Animately\giflab
```

2. Install Python dependencies:
```powershell
poetry install
```

This will install all required packages:
- Pillow (image processing)
- NumPy (numerical computing)
- Pandas (data analysis)
- OpenCV (computer vision)
- scikit-image (image analysis)
- PyTorch (machine learning)
- CLIP (AI tagging)
- And many more...

### Step 6: Enable PowerShell Scripts

```powershell
# Allow local scripts to run
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 7: Verify Installation

Run the verification script:
```powershell
.\setup_check.ps1
```

This will check:
- ✅ Python 3.11+ installation
- ✅ pip availability
- ✅ Poetry installation
- ✅ FFmpeg installation
- ✅ Gifsicle installation
- ✅ Directory structure
- ✅ Project configuration

## 🎯 Quick Start

### 1. Add Sample GIFs
```powershell
# Copy some GIF files to the raw data directory
Copy-Item "C:\path\to\your\gifs\*.gif" "data\raw\"
```

### 2. Start Analysis
```powershell
# Launch Jupyter Notebook
poetry run jupyter notebook
```

### 3. Open Analysis Notebook
- In Jupyter, navigate to `notebooks/`
- Open `01_explore_dataset.ipynb`
- Follow the step-by-step instructions

## 🔧 Troubleshooting

### "Python not found"
- Ensure Python is added to PATH during installation
- Restart PowerShell after installation
- Try: `refreshenv` (if using Chocolatey)

### "Poetry not found"
- Reinstall Poetry: `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -`
- Add to PATH manually if needed

### "FFmpeg/Gifsicle not found"
- Install via winget: `winget install Gyan.FFmpeg`
- For Gifsicle: Use Chocolatey or manual installation
- Restart PowerShell after installation
- Check PATH environment variable

### "Import errors in notebooks"
- Ensure you're in the project root directory
- Run: `poetry install` to install dependencies
- Use: `poetry run jupyter notebook` to launch Jupyter

### "Permission denied"
- Run PowerShell as Administrator for system-wide installations
- Check file permissions on the project directory

## 📊 Performance Tips

### For Small Collections (<100 GIFs)
```powershell
poetry run python -m giflab run data/raw data/ --workers 4
```

### For Medium Collections (100-1000 GIFs)
```powershell
# Run analysis first
poetry run jupyter notebook
# Then process with optimization
poetry run python -m giflab run data/raw data/ --workers 6 --resume
```

### For Large Collections (1000+ GIFs)
```powershell
# Definitely run notebooks first
poetry run python -m giflab run data/raw data/ --workers 8 --resume --use-seed-data
```

## 🎉 You're Ready!

After completing the setup:

1. **Add GIFs** to `data/raw/`
2. **Run analysis** with Jupyter notebooks
3. **Process GIFs** with the CLI
4. **Analyze results** in the generated CSV files

For detailed usage instructions, see [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)

## 📞 Getting Help

- **Setup issues**: Run `.\setup_check.ps1` and check the output
- **Usage questions**: See [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)
- **Technical details**: See [README.md](README.md)
- **Project scope**: See [PROJECT_SCOPE.md](PROJECT_SCOPE.md)

## 🏆 Why This Approach is Best Practice

### Winget Advantages:
- **Official Microsoft tool** - Built into Windows 10/11
- **Automatic PATH management** - No manual configuration needed
- **Version management** - Easy updates and rollbacks
- **Security** - Verified packages from trusted sources
- **Consistency** - Same installation method across machines

### Modern Windows Development:
- **No manual downloads** - Everything through package managers
- **Reproducible setup** - Same commands work everywhere
- **Easy maintenance** - Simple update commands
- **Professional standard** - Used by Microsoft and the community 