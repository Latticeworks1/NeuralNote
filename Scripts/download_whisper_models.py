#!/usr/bin/env python3
"""
Utility to download Whisper ONNX Runtime models for NeuralNote.

Usage:
    python Scripts/download_whisper_models.py \
        --encoder-url https://example.com/whisper_encoder.ort \
        --decoder-url https://example.com/whisper_decoder.ort

The script stores the files inside Lib/ModelData/ by default, or another
directory specified via --output.
"""

import argparse
import os
import sys
import urllib.request

DEFAULT_OUTPUT = os.path.join("Lib", "ModelData")


def download_file(url: str, destination: str) -> None:
    print(f"Downloading {url} -> {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "NeuralNote-WhisperDownloader/1.0"})
    with urllib.request.urlopen(request) as response, open(destination, "wb") as out_file:
        total = int(response.headers.get("content-length", "0"))
        downloaded = 0
        chunk_size = 1 << 14

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                print(f"\r  {downloaded}/{total} bytes ({percent:0.1f}%)", end="")
            else:
                print(f"\r  {downloaded} bytes", end="")
        print()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Download Whisper encoder/decoder ONNX models")
    parser.add_argument("--encoder-url", required=True, help="URL to whisper_encoder.ort")
    parser.add_argument("--decoder-url", required=True, help="URL to whisper_decoder.ort")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Destination directory (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])

    os.makedirs(args.output, exist_ok=True)

    encoder_path = os.path.join(args.output, "whisper_encoder.ort")
    decoder_path = os.path.join(args.output, "whisper_decoder.ort")

    download_file(args.encoder_url, encoder_path)
    download_file(args.decoder_url, decoder_path)

    print("Download complete. Rebuild NeuralNote to embed the new models.")


if __name__ == "__main__":
    main()
