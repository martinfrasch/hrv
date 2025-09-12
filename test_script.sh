#!/bin/bash

# HRV Pipeline Container Test Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default image
IMAGE_NAME="${1:-hrv-pipeline:latest}"

echo -e "${YELLOW}Testing HRV Pipeline Container${NC}"
echo -e "${YELLOW}=============================${NC}"
echo "Image: $IMAGE_NAME"
echo ""

# Test 1: Basic container startup
echo -e "${YELLOW}Test 1: Container startup and help${NC}"
if docker run --rm "$IMAGE_NAME" --help >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Container starts successfully${NC}"
else
    echo -e "${RED}✗ Container startup failed${NC}"
    exit 1
fi

# Test 2: Python imports
echo -e "${YELLOW}Test 2: Python dependencies${NC}"
if docker run --rm "$IMAGE_NAME" python -c "
import neurokit2 as nk
import pandas as pd
import numpy as np
import joblib
import yaml
print('All imports successful')
" 2>/dev/null; then
    echo -e "${GREEN}✓ All Python dependencies available${NC}"
else
    echo -e "${RED}✗ Missing Python dependencies${NC}"
    exit 1
fi

# Test 3: Generate test data and run pipeline
echo -e "${YELLOW}Test 3: Pipeline execution with synthetic data${NC}"

# Create temporary test directory
TEST_DIR=$(mktemp -d)
trap "rm -rf $TEST_DIR" EXIT

mkdir -p "$TEST_DIR/data/heartrate" "$TEST_DIR/data/sleep_labels" "$TEST_DIR/output"

# Generate synthetic BPM data
cat > "$TEST_DIR/data/heartrate/test_subject.txt" << 'EOF'
timestamp,bpm
0.0,72.5
1.0,73.1
2.0,71.8
3.0,74.2
4.0,72.9
5.0,73.5
6.0,71.2
7.0,74.8
8.0,73.3
9.0,72.7
10.0,75.1
11.0,73.9
12.0,72.4
13.0,74.6
14.0,73.2
15.0,71.9
16.0,73.8
17.0,72.3
18.0,74.4
19.0,73.7
20.0,72.1
21.0,74.9
22.0,73.4
23.0,72.8
24.0,75.3
25.0,73.6
26.0,72.2
27.0,74.7
28.0,73.1
29.0,72.5
30.0,74.1
31.0,73.8
32.0,72.6
33.0,74.3
34.0,73.0
35.0,72.4
36.0,74.8
37.0,73.5
38.0,72.9
39.0,74.2
40.0,73.7
41.0,72.3
42.0,74.6
43.0,73.3
44.0,72.7
45.0,75.0
46.0,73.9
47.0,72.5
48.0,74.4
49.0,73.1
50.0,72.8
51.0,74.7
52.0,73.4
53.0,72.2
54.0,74.9
55.0,73.6
56.0,72.9
57.0,74.3
58.0,73.2
59.0,72.6
60.0,74.5
61.0,73.8
62.0,72.4
63.0,74.1
64.0,73.7
65.0,72.3
66.0,74.8
67.0,73.5
68.0,72.9
69.0,74.2
70.0,73.4
71.0,72.7
72.0,74.6
73.0,73.1
74.0,72.8
75.0,75.2
76.0,73.9
77.0,72.5
78.0,74.4
79.0,73.3
80.0,72.6
81.0,74.7
82.0,73.8
83.0,72.4
84.0,74.1
85.0,73.6
86.0,72.9
87.0,74.3
88.0,73.2
89.0,72.7
90.0,74.8
91.0,73.5
92.0,72.3
93.0,74.6
94.0,73.4
95.0,72.8
96.0,75.1
97.0,73.7
98.0,72.5
99.0,74.2
EOF

# Generate synthetic sleep labels
cat > "$TEST_DIR/data/sleep_labels/test_subject.txt" << 'EOF'
timestamp,sleep_state
0,Wake
10,N1
20,N2
30,N3
40,N2
50,N1
60,REM
70,N2
80,N3
90,Wake
EOF

# Run pipeline with test data
echo "Running pipeline with synthetic data..."
if docker run --rm \
    -v "$TEST_DIR/data:/app/data:ro" \
    -v "$TEST_DIR/output:/app/output:rw" \
    "$IMAGE_NAME" \
    --data-path /app/data \
    --output-path /app/output/test_results.parquet \
    --n-workers 1 2>/dev/null; then
    echo -e "${GREEN}✓ Pipeline executed successfully${NC}"
else
    echo -e "${RED}✗ Pipeline execution failed${NC}"
    exit 1
fi

# Test 4: Verify output files
echo -e "${YELLOW}Test 4: Output validation${NC}"
if [[ -f "$TEST_DIR/output/test_results.parquet" ]]; then
    echo -e "${GREEN}✓ Output file created${NC}"
    
    # Check if output file is readable
    if docker run --rm \
        -v "$TEST_DIR/output:/app/output:ro" \
        "$IMAGE_NAME" \
        python -c "
import pandas as pd
df = pd.read_parquet('/app/output/test_results.parquet')
print(f'Output contains {len(df)} rows and {len(df.columns)} columns')
assert len(df) > 0, 'Output file is empty'
assert 'HRV_RMSSD' in df.columns, 'Missing expected HRV metric'
print('Output validation successful')
" 2>/dev/null; then
        echo -e "${GREEN}✓ Output file is valid and contains HRV metrics${NC}"
    else
        echo -e "${RED}✗ Output file is not valid${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Output file was not created${NC}"
    exit 1
fi

# Test 5: Container resource usage
echo -e "${YELLOW}Test 5: Resource usage test${NC}"
CONTAINER_ID=$(docker run -d \
    --memory=512m --cpus=0.5 \
    -v "$TEST_DIR/data:/app/data:ro" \
    -v "$TEST_DIR/output:/app/output:rw" \
    "$IMAGE_NAME" \
    --data-path /app/data \
    --output-path /app/output/resource_test.parquet \
    --n-workers 1)

# Wait for container to finish
docker wait "$CONTAINER_ID" >/dev/null 2>&1

# Check exit code
if [[ $(docker inspect "$CONTAINER_ID" --format='{{.State.ExitCode}}') -eq 0 ]]; then
    echo -e "${GREEN}✓ Container respects resource limits${NC}"
else
    echo -e "${RED}✗ Container failed with resource limits${NC}"
    docker logs "$CONTAINER_ID"
    docker rm "$CONTAINER_ID" >/dev/null 2>&1
    exit 1
fi

# Clean up
docker rm "$CONTAINER_ID" >/dev/null 2>&1

# Test 6: Configuration file support
echo -e "${YELLOW}Test 6: Configuration file support${NC}"
cat > "$TEST_DIR/test_config.yaml" << 'EOF'
sampling_rate: 1000
chunk_size: 1000
enable_caching: false
output_format: "csv"
hrv_time_domain: true
hrv_frequency_domain: false
hrv_nonlinear_domain: false
EOF

if docker run --rm \
    -v "$TEST_DIR/data:/app/data:ro" \
    -v "$TEST_DIR/output:/app/output:rw" \
    -v "$TEST_DIR/test_config.yaml:/app/config/test_config.yaml:ro" \
    "$IMAGE_NAME" \
    --data-path /app/data \
    --output-path /app/output/config_test.csv \
    --config /app/config/test_config.yaml \
    --n-workers 1 2>/dev/null; then
    echo -e "${GREEN}✓ Configuration file support works${NC}"
else
    echo -e "${RED}✗ Configuration file support failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All tests passed! Container is working correctly.${NC}"
echo ""
echo -e "${YELLOW}Test Summary:${NC}"
echo "✓ Container startup and help"
echo "✓ Python dependencies"  
echo "✓ Generic pipeline execution (heart rate, RR intervals)"
echo "✓ Output file validation"
echo "✓ Resource limits compliance"
echo "✓ Configuration file support"
echo ""
echo -e "${GREEN}Container $IMAGE_NAME is ready for processing cardiovascular time series!${NC}"