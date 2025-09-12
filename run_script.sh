#!/bin/bash

# HRV Pipeline Container Run Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="generic-hrv-pipeline:latest"
DATA_DIR="./data"
OUTPUT_DIR="./output"
CACHE_DIR="./cache"
CONFIG_FILE="./config/hrv_config.yaml"
N_WORKERS="4"
CONTAINER_NAME="hrv-pipeline-$(date +%s)"
INPUT_TYPE="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image|-i)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --data|-d)
            DATA_DIR="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cache|-c)
            CACHE_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --workers|-w)
            N_WORKERS="$2"
            shift 2
            ;;
        --name|-n)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --interactive|-it)
            INTERACTIVE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [-- PIPELINE_ARGS]"
            echo "Options:"
            echo "  --image, -i IMAGE     Docker image to run (default: hrv-pipeline:latest)"
            echo "  --data, -d DIR        Data directory (default: ./data)"
            echo "  --output, -o DIR      Output directory (default: ./output)"
            echo "  --cache, -c DIR       Cache directory (default: ./cache)"
            echo "  --config FILE         Configuration file (default: ./config/hrv_config.yaml)"
            echo "  --workers, -w N       Number of workers (default: 4)"
            echo "  --name, -n NAME       Container name"
            echo "  --interactive, -it    Run in interactive mode"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --data /path/to/data --output /path/to/results"
            echo "  $0 --workers 8 -- --data-path /app/data --output-path /app/output/results.parquet"
            echo "  $0 --interactive  # For debugging"
            exit 0
            ;;
        --)
            shift
            PIPELINE_ARGS="$@"
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Running HRV Pipeline Container${NC}"
echo -e "${YELLOW}==============================${NC}"
echo "Image: $IMAGE_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Cache directory: $CACHE_DIR"
echo "Workers: $N_WORKERS"
echo ""

# Validate directories exist
for dir in "$DATA_DIR" "$(dirname "$OUTPUT_DIR")" "$(dirname "$CACHE_DIR")"; do
    if [[ ! -d "$dir" ]]; then
        echo -e "${YELLOW}Creating directory: $dir${NC}"
        mkdir -p "$dir"
    fi
done

# Create required subdirectories in data directory
mkdir -p "$DATA_DIR/heartrate" "$DATA_DIR/sleep_labels"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# Check if data directory has files
if [[ ! -d "$DATA_DIR/heartrate" ]] || [[ -z "$(ls -A "$DATA_DIR/heartrate" 2>/dev/null)" ]]; then
    echo -e "${YELLOW}Warning: No heartrate data found in $DATA_DIR/heartrate${NC}"
fi

# Build docker run command
DOCKER_ARGS=(
    --rm
    --name "$CONTAINER_NAME"
    -v "$(realpath "$DATA_DIR"):/app/data:ro"
    -v "$(realpath "$OUTPUT_DIR"):/app/output:rw"
    -v "$(realpath "$CACHE_DIR"):/app/cache:rw"
    -e "HRV_N_WORKERS=$N_WORKERS"
    -e "PYTHONPATH=/app"
)

# Add config file if it exists
if [[ -f "$CONFIG_FILE" ]]; then
    DOCKER_ARGS+=(-v "$(realpath "$CONFIG_FILE"):/app/config/hrv_config.yaml:ro")
    echo "Using config file: $CONFIG_FILE"
fi

# Add interactive mode if requested
if [[ "$INTERACTIVE" == "true" ]]; then
    DOCKER_ARGS+=(-it)
    DOCKER_ARGS+=(--entrypoint /bin/bash)
    IMAGE_ARGS=""
else
    # Set default pipeline arguments if none provided
    if [[ -z "$PIPELINE_ARGS" ]]; then
        PIPELINE_ARGS="--data-path /app/data --output-path /app/output/hrv_results --n-workers $N_WORKERS"
        if [[ -f "$CONFIG_FILE" ]]; then
            PIPELINE_ARGS="$PIPELINE_ARGS --config /app/config/hrv_config.yaml"
        fi
    fi
    IMAGE_ARGS="$PIPELINE_ARGS"
fi

echo -e "${YELLOW}Running container...${NC}"
echo "Command: docker run ${DOCKER_ARGS[*]} $IMAGE_NAME $IMAGE_ARGS"
echo ""

# Run the container
if docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" $IMAGE_ARGS; then
    echo ""
    echo -e "${GREEN}✓ Pipeline completed successfully${NC}"
    echo -e "${YELLOW}Results available in: $OUTPUT_DIR${NC}"
    
    # Show output files
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "Output files:"
        ls -la "$OUTPUT_DIR"
    fi
else
    echo ""
    echo -e "${RED}✗ Pipeline failed${NC}"
    exit 1
fi