#!/bin/bash
set -e

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate hurricane-forecast

# Set up environment variables
export PYTHONPATH=/app:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Create necessary directories if they don't exist
mkdir -p /data/hurricane-forecast/{hurdat2,ibtracs,era5,models,checkpoints,cache,logs,mlruns}

# Check GPU availability
echo "=== GPU Status ==="
nvidia-smi || echo "No GPU detected"
echo ""

# Check Python environment
echo "=== Python Environment ==="
which python
python --version
echo ""

# Test PyTorch GPU support
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

# Function to wait for services
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    
    echo "Waiting for $service at $host:$port..."
    while ! nc -z $host $port; do
        sleep 1
    done
    echo "$service is ready!"
}

# Handle different commands
case "$1" in
    # Development mode - just run bash
    bash)
        exec bash
        ;;
    
    # Jupyter Lab
    jupyter)
        echo "Starting Jupyter Lab..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
            --NotebookApp.token='' --NotebookApp.password=''
        ;;
    
    # API server
    api)
        echo "Starting FastAPI server..."
        exec uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload
        ;;
    
    # Training
    train)
        echo "Starting training..."
        shift
        exec python scripts/train_model.py "$@"
        ;;
    
    # Evaluation
    evaluate)
        echo "Starting evaluation..."
        shift
        exec python scripts/evaluate.py "$@"
        ;;
    
    # MLflow server
    mlflow)
        echo "Starting MLflow server..."
        exec mlflow server \
            --backend-store-uri file:///data/hurricane-forecast/mlruns \
            --default-artifact-root file:///data/hurricane-forecast/mlruns \
            --host 0.0.0.0 \
            --port 5000
        ;;
    
    # TensorBoard
    tensorboard)
        echo "Starting TensorBoard..."
        exec tensorboard --logdir=/data/hurricane-forecast/logs --host=0.0.0.0 --port=6006
        ;;
    
    # Data setup
    setup)
        echo "Running data setup..."
        shift
        exec python scripts/setup_data.py "$@"
        ;;
    
    # Run tests
    test)
        echo "Running tests..."
        shift
        exec pytest tests/ "$@"
        ;;
    
    # Custom command
    *)
        exec "$@"
        ;;
esac
