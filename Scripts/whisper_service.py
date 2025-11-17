#!/usr/bin/env python3
"""
NeuralNote Whisper Transcription Service

Provides a local HTTP API for speech-to-text transcription using
Hugging Face Transformers and OpenAI's Whisper model.

The service runs locally and provides endpoints for:
- Health checks
- Audio transcription with word-level timestamps
- Language detection
- Model info

Usage:
    python Scripts/whisper_service.py --port 8765 --model openai/whisper-large-v3-turbo
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
whisper_pipeline = None
model_info = {}


def initialize_model(model_id: str, device: str = "auto", model_dir: Optional[str] = None) -> None:
    """Initialize the Whisper model and pipeline."""
    global whisper_pipeline, model_info

    logger.info(f"Initializing Whisper model: {model_id}")
    if model_dir:
        logger.info(f"Using model directory: {model_dir}")

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Using device: {device}, dtype: {torch_dtype}")

    try:
        # Prepare kwargs for loading from custom directory
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }

        if model_dir:
            load_kwargs["cache_dir"] = model_dir

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            **load_kwargs
        )
        model.to(device)

        processor_kwargs = {}
        if model_dir:
            processor_kwargs["cache_dir"] = model_dir

        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps="word"
        )

        model_info = {
            "model_id": model_id,
            "device": str(device),
            "dtype": str(torch_dtype),
            "sample_rate": processor.feature_extractor.sampling_rate
        }

        logger.info("Model initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if whisper_pipeline is None:
        return jsonify({"status": "error", "message": "Model not initialized"}), 503

    return jsonify({
        "status": "healthy",
        "model": model_info
    })


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio to text with word-level timestamps.

    Expected request body:
    {
        "audio": [float array of audio samples at 16kHz],
        "language": "en" (optional, auto-detect if not provided),
        "task": "transcribe" (or "translate")
    }

    Returns:
    {
        "text": "full transcription",
        "words": [
            {"text": "word", "start": 0.0, "end": 0.5, "confidence": 0.95},
            ...
        ]
    }
    """
    if whisper_pipeline is None:
        return jsonify({"error": "Model not initialized"}), 503

    try:
        data = request.get_json()

        if 'audio' not in data:
            return jsonify({"error": "Missing 'audio' field"}), 400

        audio_array = np.array(data['audio'], dtype=np.float32)

        # Validate sample rate (should be 16kHz)
        expected_sr = model_info.get('sample_rate', 16000)
        if 'sample_rate' in data and data['sample_rate'] != expected_sr:
            logger.warning(f"Audio sample rate {data['sample_rate']} doesn't match expected {expected_sr}")

        # Prepare generation kwargs
        generate_kwargs = {
            "max_new_tokens": 448,
            "task": data.get('task', 'transcribe'),
            "return_timestamps": "word"
        }

        # Add language if specified
        if 'language' in data and data['language']:
            generate_kwargs['language'] = data['language']

        # Run transcription
        logger.info(f"Transcribing audio: {len(audio_array)} samples, language={generate_kwargs.get('language', 'auto')}")

        result = whisper_pipeline(
            {"array": audio_array, "sampling_rate": expected_sr},
            generate_kwargs=generate_kwargs
        )

        # Extract word-level timestamps
        words = []
        if 'chunks' in result:
            for chunk in result['chunks']:
                if 'timestamp' in chunk and chunk['timestamp'] is not None:
                    start_time, end_time = chunk['timestamp']
                    words.append({
                        "text": chunk['text'].strip(),
                        "start": float(start_time) if start_time is not None else 0.0,
                        "end": float(end_time) if end_time is not None else 0.0,
                        "confidence": 1.0  # Transformers doesn't provide confidence scores
                    })

        response = {
            "text": result.get('text', ''),
            "words": words
        }

        logger.info(f"Transcription complete: {len(words)} words")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model."""
    if whisper_pipeline is None:
        return jsonify({"error": "Model not initialized"}), 503

    return jsonify(model_info)


def parse_args():
    parser = argparse.ArgumentParser(
        description='NeuralNote Whisper Transcription Service'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='openai/whisper-large-v3-turbo',
        help='Hugging Face model ID or path to local model directory (default: openai/whisper-large-v3-turbo)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Directory containing pre-downloaded models (e.g., ~/.cache/huggingface/hub). If specified, models will be loaded from here instead of downloading.'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='Port to run the service on (default: 8765)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use: auto, cpu, cuda:0, etc. (default: auto)'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available Whisper models from Hugging Face and exit'
    )
    return parser.parse_args()


def list_available_models():
    """List commonly used Whisper models."""
    models = [
        "openai/whisper-tiny",
        "openai/whisper-tiny.en",
        "openai/whisper-base",
        "openai/whisper-base.en",
        "openai/whisper-small",
        "openai/whisper-small.en",
        "openai/whisper-medium",
        "openai/whisper-medium.en",
        "openai/whisper-large-v2",
        "openai/whisper-large-v3",
        "openai/whisper-large-v3-turbo",
        "distil-whisper/distil-large-v2",
        "distil-whisper/distil-large-v3",
        "distil-whisper/distil-medium.en",
        "distil-whisper/distil-small.en"
    ]

    print("\n" + "=" * 60)
    print("Available Whisper Models (Hugging Face)")
    print("=" * 60)
    print("\nOpenAI Whisper (original):")
    for model in models[:11]:
        print(f"  - {model}")
    print("\nDistil-Whisper (faster, smaller):")
    for model in models[11:]:
        print(f"  - {model}")
    print("\nUsage:")
    print("  python Scripts/whisper_service.py --model <model_id>")
    print("\nExample:")
    print("  python Scripts/whisper_service.py --model openai/whisper-large-v3-turbo")
    print("\nFor local models:")
    print("  python Scripts/whisper_service.py --model /path/to/model --model-dir /cache/dir")
    print()


def main():
    args = parse_args()

    if args.list_models:
        list_available_models()
        sys.exit(0)

    logger.info("=" * 60)
    logger.info("NeuralNote Whisper Transcription Service")
    logger.info("=" * 60)

    # Initialize model
    try:
        initialize_model(args.model, args.device, args.model_dir)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)

    # Run service
    logger.info(f"Starting service on {args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
