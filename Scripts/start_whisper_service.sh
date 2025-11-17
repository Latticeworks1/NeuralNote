#!/bin/bash
# Launcher script for NeuralNote Whisper Service
# Usage: ./Scripts/start_whisper_service.sh [model_id] [port]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_ID="${1:-openai/whisper-large-v3-turbo}"
PORT="${2:-8765}"

echo "================================================================"
echo "NeuralNote Whisper Service Launcher"
echo "================================================================"
echo "Model: $MODEL_ID"
echo "Port: $PORT"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
VENV_DIR="$PROJECT_ROOT/venv-whisper"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at: $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install/upgrade dependencies
echo "Checking dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements-whisper-service.txt"

echo ""
echo "Starting Whisper service..."
echo "Press Ctrl+C to stop"
echo ""

# Run the service
python3 "$SCRIPT_DIR/whisper_service.py" \
    --model "$MODEL_ID" \
    --port "$PORT" \
    --device auto
