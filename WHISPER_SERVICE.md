# Whisper Service Setup Guide

This guide explains how to use the Hugging Face Transformers-based Whisper service for text transcription in NeuralNote.

## Overview

NeuralNote supports two backends for text transcription:

1. **ONNX Runtime** (local, embedded models) - Default, runs entirely in C++
2. **HTTP Service** (Hugging Face Transformers via Python) - Better accuracy, more features, easier model selection

The HTTP Service option provides access to the full ecosystem of Whisper models on Hugging Face, including:
- All official OpenAI Whisper variants (tiny, base, small, medium, large-v2, large-v3, large-v3-turbo)
- Distil-Whisper models (faster, smaller alternatives)
- Fine-tuned models for specific languages/domains

## Quick Start

### 1. Install Python Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv-whisper
source venv-whisper/bin/activate  # On Windows: venv-whisper\Scripts\activate

# Install requirements
pip install -r Scripts/requirements-whisper-service.txt
```

### 2. Start the Service

**Option A: Using the launcher script (recommended)**
```bash
./Scripts/start_whisper_service.sh
```

**Option B: Manual start**
```bash
python3 Scripts/whisper_service.py --model openai/whisper-large-v3-turbo
```

### 3. Launch NeuralNote

The plugin will automatically detect and use the service if it's running on `http://127.0.0.1:8765`.

## Model Selection

### List Available Models

```bash
python3 Scripts/whisper_service.py --list-models
```

### Using Different Models

Start the service with a specific model:

```bash
# Latest turbo model (fastest large model)
python3 Scripts/whisper_service.py --model openai/whisper-large-v3-turbo

# Smaller, faster models
python3 Scripts/whisper_service.py --model openai/whisper-small
python3 Scripts/whisper_service.py --model openai/whisper-tiny

# Distil-Whisper (6x faster)
python3 Scripts/whisper_service.py --model distil-whisper/distil-large-v3

# English-only models (more accurate for English)
python3 Scripts/whisper_service.py --model openai/whisper-medium.en
```

### Using Local/Cached Models

If you already have Hugging Face models downloaded:

```bash
# Use models from custom cache directory
python3 Scripts/whisper_service.py \
    --model openai/whisper-large-v3-turbo \
    --model-dir ~/.cache/huggingface/hub

# Or use a local model directory
python3 Scripts/whisper_service.py \
    --model /path/to/my/local/whisper/model
```

## Advanced Configuration

### Custom Port

```bash
python3 Scripts/whisper_service.py --port 9000
```

If using a custom port, set the `NEURALNOTE_WHISPER_SERVICE_URL` environment variable:

```bash
export NEURALNOTE_WHISPER_SERVICE_URL=http://127.0.0.1:9000
```

### Device Selection

```bash
# Automatic (default: GPU if available, else CPU)
python3 Scripts/whisper_service.py --device auto

# Force CPU
python3 Scripts/whisper_service.py --device cpu

# Specific GPU
python3 Scripts/whisper_service.py --device cuda:0
```

### Performance Optimization

For faster inference, install Flash Attention 2 (if your GPU supports it):

```bash
pip install flash-attn --no-build-isolation
```

The service will automatically use Flash Attention if available.

## API Endpoints

The service provides the following HTTP endpoints:

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": {
    "model_id": "openai/whisper-large-v3-turbo",
    "device": "cuda:0",
    "dtype": "torch.float16",
    "sample_rate": 16000
  }
}
```

### POST /transcribe

Transcribe audio to text.

**Request:**
```json
{
  "audio": [/* float array of audio samples at 16kHz */],
  "language": "en",  // Optional, auto-detect if omitted
  "task": "transcribe"  // Or "translate" for translation to English
}
```

**Response:**
```json
{
  "text": "full transcription",
  "words": [
    {
      "text": "hello",
      "start": 0.0,
      "end": 0.5,
      "confidence": 1.0
    },
    // ...
  ]
}
```

### GET /info

Get model information.

## Troubleshooting

### Service Won't Start

**Problem:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:** Install dependencies:
```bash
pip install -r Scripts/requirements-whisper-service.txt
```

**Problem:** `Model not found` or download errors

**Solution:** Ensure you have internet connection for first-time model download, or specify a local model with `--model-dir`.

### NeuralNote Not Detecting Service

1. Verify service is running:
   ```bash
   curl http://127.0.0.1:8765/health
   ```

2. Check the service logs for errors

3. Ensure no firewall is blocking port 8765

### Poor Performance

1. Use smaller models (tiny, base, small) for faster inference
2. Consider Distil-Whisper models (6x faster)
3. Ensure GPU is being used: check service logs for `device: cuda:0`
4. Install Flash Attention 2 for additional speedup

## Backend Selection

NeuralNote uses automatic backend selection:

1. If HTTP service is running → Use HTTP service (Hugging Face Transformers)
2. If ONNX models are available → Use ONNX Runtime
3. If neither available → Show placeholder message

You can check which backend is active in the NeuralNote debug output.

## Model Comparison

| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|---------|
| whisper-tiny | 39M | 32x | Good | 1GB |
| whisper-base | 74M | 16x | Better | 1GB |
| whisper-small | 244M | 6x | Very Good | 2GB |
| whisper-medium | 769M | 2x | Excellent | 5GB |
| whisper-large-v3-turbo | 809M | 1.5x | Best | 6GB |
| distil-large-v3 | 756M | 6x | Excellent | 4GB |

Speed is relative to whisper-large-v3. Distil-Whisper provides near-large accuracy at small model speed.

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Whisper Model Card](https://huggingface.co/openai/whisper-large-v3-turbo)
- [Distil-Whisper](https://github.com/huggingface/distil-whisper)
