#!/bin/bash

# HRV Pipeline Container Build Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="generic-hrv-pipeline"
TAG="latest"
BUILD_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag|-t)
            TAG="$2"
            shift 2
            ;;
        --name|-n)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --no-cache)
            BUILD_ARGS="$BUILD_ARGS --no-cache"
            shift
            ;;
        --dev)
            TAG="dev"
            BUILD_ARGS="$BUILD_ARGS --target builder"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --tag, -t TAG          Set image tag (default: latest)"
            echo "  --name, -n NAME        Set image name (default: hrv-pipeline)"
            echo "  --no-cache            Build without using cache"
            echo "  --dev                 Build development version"
            echo "  --help, -h            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo -e "${YELLOW}Building HRV Pipeline Docker Image${NC}"
echo -e "${YELLOW}===================================${NC}"
echo "Image: $FULL_IMAGE_NAME"
echo "Build args: $BUILD_ARGS"
echo ""

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    echo -e "${RED}Error: Dockerfile not found in current directory${NC}"
    exit 1
fi

# Check if required files exist
required_files=("requirements.txt" "src/hrv_pipeline.py")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}Error: Required file $file not found${NC}"
        exit 1
    fi
done

# Build the Docker image
echo -e "${YELLOW}Starting Docker build...${NC}"
if docker build $BUILD_ARGS -t "$FULL_IMAGE_NAME" .; then
    echo -e "${GREEN}✓ Successfully built $FULL_IMAGE_NAME${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Show image size
echo ""
echo -e "${YELLOW}Image Information:${NC}"
docker images "$IMAGE_NAME" | head -2

# Optional: Run basic tests
if [[ -f "scripts/test.sh" ]]; then
    echo ""
    read -p "Run basic tests? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash scripts/test.sh "$FULL_IMAGE_NAME"
    fi
fi

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run the container: bash scripts/run.sh"
echo "2. Or use docker-compose: docker-compose up"
echo "3. For help: docker run --rm $FULL_IMAGE_NAME --help"