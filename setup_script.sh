#!/bin/bash

# HRV Pipeline Container Setup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}HRV Pipeline Container Setup${NC}"
echo -e "${BLUE}============================${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command_exists docker; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
else
    echo -e "${GREEN}✓ Docker is installed${NC}"
fi

if ! command_exists docker-compose; then
    echo -e "${YELLOW}⚠ Docker Compose is not installed (optional)${NC}"
    echo "  Install Docker Compose for easier orchestration: https://docs.docker.com/compose/install/"
else
    echo -e "${GREEN}✓ Docker Compose is installed${NC}"
fi

# Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}✗ Docker daemon is not running${NC}"
    echo "Please start Docker daemon"
    exit 1
else
    echo -e "${GREEN}✓ Docker daemon is running${NC}"
fi

echo ""

# Create directory structure
echo -e "${YELLOW}Creating directory structure...${NC}"

directories=(
    "data/heartrate"
    "data/sleep_labels"
    "output"
    "cache" 
    "logs"
    "config"
    "src/utils"
    "tests/test_data"
    "scripts"
)

for dir in "${directories[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        echo -e "${GREEN}✓ Created directory: $dir${NC}"
    else
        echo -e "${YELLOW}- Directory exists: $dir${NC}"
    fi
done

echo ""

# Copy optimized pipeline code
echo -e "${YELLOW}Setting up application code...${NC}"

# Create __init__.py files
touch src/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

echo -e "${GREEN}✓ Created Python package structure${NC}"

# Create .dockerignore file
cat > .dockerignore << 'EOF'
# Git
.git/
.gitignore

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Data directories (will be mounted as volumes)
data/
output/
cache/
logs/

# Documentation
docs/
*.md
!README.md

# Development scripts
dev/
notebooks/
EOF

echo -e "${GREEN}✓ Created .dockerignore${NC}"

# Create sample configuration
if [[ ! -f "config/hrv_config.yaml" ]]; then
    cp config/docker_config.yaml config/hrv_config.yaml
    echo -e "${GREEN}✓ Created default configuration${NC}"
else
    echo -e "${YELLOW}- Configuration file already exists${NC}"
fi

# Make scripts executable
if [[ -d "scripts" ]]; then
    chmod +x scripts/*.sh 2>/dev/null || true
    echo -e "${GREEN}✓ Made scripts executable${NC}"
fi

echo ""

# Create README for data directory
cat > data/README.md << 'EOF'
# Data Directory

Place your cardiovascular time series data here organized by signal type:

```
data/
├── ecg/                    # Raw ECG signals
│   ├── patient_001.csv
│   ├── recording_*.h5
│   └── ecg_data.txt
├── heartrate/              # Heart rate (BPM) data  
│   ├── wearable_hr.csv
│   ├── monitor_bpm.xlsx
│   └── subject_*.txt
├── rr_intervals/           # RR interval time series
│   ├── intervals_ms.csv
│   ├── rri_data.parquet
│   └── beat_intervals.txt
└── mixed/                  # Unknown/mixed types (auto-detect)
    ├── unknown_signal.csv
    └── mystery_data.txt
```

## Signal Types & Formats

### ECG Signals (Raw Electrocardiogram)
- **Format**: Time series with voltage/amplitude values
- **Units**: mV, µV, or normalized
- **Sampling Rate**: Specify with --sampling-rate parameter
- **Example**: time,ecg\n0.000,-0.1\n0.004,0.2\n...

### Heart Rate Data (BPM)
- **Format**: Beat-per-minute values over time
- **Units**: beats/minute (BPM)
- **Source**: Wearables, fitness trackers, monitors
- **Example**: timestamp,bpm\n0,72.5\n1,73.1\n...

### RR Intervals
- **Format**: Inter-beat interval durations
- **Units**: milliseconds (ms) or seconds (s)
- **Source**: Pre-processed from ECG or pulse sensors
- **Example**: time,rr_ms\n0,833.2\n1,822.1\n...

## Supported File Formats
- **CSV**: Standard comma-separated values
- **TXT/TSV**: Tab or space-separated text
- **Excel**: .xlsx, .xls files
- **Parquet**: Columnar format (fastest)
- **HDF5**: Hierarchical data format

## File Structure Requirements
All files must have **at least 2 columns**:
- Column 1: Time/timestamp (seconds, samples, datetime)
- Column 2: Signal values (ECG, BPM, or RR intervals)

## Auto-Detection
The pipeline can automatically detect your signal type if you're unsure.
Place files in `data/mixed/` for automatic detection.
EOF

echo -e "${GREEN}✓ Created data directory documentation${NC}"

echo ""
echo -e "${GREEN}Setup completed successfully!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Place your data files in the data/ directory"
echo "2. Build the container: bash scripts/build.sh"  
echo "3. Run the pipeline: bash scripts/run.sh"
echo ""
echo -e "${YELLOW}Alternative using Docker Compose:${NC}"
echo "1. Place your data files in the data/ directory"
echo "2. Run: docker-compose up"
echo ""
echo -e "${YELLOW}For help:${NC}"
echo "- Build script: bash scripts/build.sh --help"
echo "- Run script: bash scripts/run.sh --help"
echo "- Pipeline help: docker run --rm hrv-pipeline:latest --help"