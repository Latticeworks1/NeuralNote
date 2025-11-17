# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NeuralNote is a cross-platform audio plugin (VST3/AU/Standalone) that performs real-time Audio-to-MIDI transcription using Spotify's Basic Pitch model. The plugin is built with JUCE and uses RTNeural for CNN inference and ONNXRuntime for feature extraction (Constant-Q Transform + Harmonic Stacking).

## Build Commands

### First-Time Setup

Clone with submodules:
```bash
git clone --recurse-submodules --shallow-submodules https://github.com/DamRsn/NeuralNote
```

### macOS

```bash
./build.sh
```

This script:
- Downloads pre-built ONNXRuntime static library
- Extracts the optimized ONNX model to `Lib/ModelData/features_model.ort`
- Configures CMake in Release mode with tests enabled
- Builds all targets (Standalone, VST3, AU)
- Runs unit tests
- Output: `./build/NeuralNote_artefacts/Release/Standalone/NeuralNote.app/Contents/MacOS/NeuralNote`

### Windows

For Visual Studio 2022 (MSVC 19.35.x):
```cmd
.\build.bat
```

For other MSVC versions, manually build ONNXRuntime first (see README.md for detailed steps), then run `.\build.bat`.

### Linux

```bash
./build.sh
```

Tests are disabled on Linux by default. Output: `./build/NeuralNote_artefacts/Release/Standalone/NeuralNote`

### IDE Development

After running the build script once, you can load the project in any CMake-compatible IDE (CLion, Visual Studio, VSCode). The build script only needs to run once to download ONNXRuntime dependencies.

### Testing

Run unit tests (macOS/Windows only):
```bash
./build/Tests/UnitTests_artefacts/Release/UnitTests
```

### Build Options

Configure with CMake options:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DUniversalBinary=ON \      # macOS universal binary (x86_64 + arm64)
  -DLTO=ON \                  # Link-time optimization
  -DBUILD_UNIT_TESTS=ON       # Enable unit tests
```

## Architecture

### High-Level Structure

```
NeuralNote/
├── NeuralNote/           # Plugin-specific code (JUCE plugin wrapper + UI)
│   ├── PluginSources/    # JUCE plugin processor (PluginProcessor, PluginEditor)
│   ├── Source/           # Application logic (managers, controllers, components)
│   └── Assets/           # UI resources (fonts, icons, images)
├── Lib/                  # Core transcription engine (reusable library)
│   ├── Model/            # BasicPitch implementation (CNN, Features, Notes)
│   ├── Components/       # UI components (Keyboard, Knobs, Sliders)
│   ├── DSP/              # Audio processing (Resampler)
│   ├── MidiPostProcessing/ # MIDI quantization and scaling
│   ├── Player/           # Audio/MIDI playback
│   ├── Utils/            # Utility functions
│   └── ModelData/        # Model weights (.json files for CNN, .ort for features)
├── Tests/                # Unit tests
└── ThirdParty/           # External dependencies (JUCE, RTNeural, ONNXRuntime)
```

### Key Components

**Plugin Layer (`NeuralNote/`)**:
- `PluginProcessor`: Main audio plugin processor, manages state machine (Empty → Recording → Processing → Populated)
- `PluginEditor`: Plugin UI, hosts the main view
- `TranscriptionManager`: Coordinates transcription workflow on a background thread pool
- `SourceAudioManager`: Manages input audio recording and file loading (.wav, .aiff, .flac, .mp3, .ogg)
- `Player`: Handles playback of source audio and synthesized MIDI
- `SynthController`: Generates audio from MIDI notes for preview

**Transcription Engine (`Lib/Model/`)**:
- `BasicPitch`: Main API for transcription. Call `transcribeToMIDI(audio, numSamples)` → get `getNoteEvents()`
- `Features`: Computes CQT + Harmonic Stacking features using ONNXRuntime
- `BasicPitchCNN`: Runs CNN inference using RTNeural (4 sequential 2D conv models)
- `Notes`: Converts posteriorgrams (note/onset/contour probabilities) to MIDI note events

**Pipeline**: Raw Audio (22.05kHz) → Features (CQT) → BasicPitchCNN → Posteriorgrams → Notes → MIDI Events

**UI Components (`NeuralNote/Source/Components/`)**:
- `NeuralNoteMainView`: Main container view
- `CombinedAudioMidiRegion`: Waveform + piano roll display
- `PianoRoll`: Piano roll visualization with zoom/scroll
- `AudioRegion`: Waveform display
- `MidiFileDrag`: Drag-and-drop MIDI export
- Custom widgets: `Keyboard`, `Knob`, `MinMaxNoteSlider`, `QuantizeForceSlider`

### State Management

The plugin uses a state machine (see `State` enum in `PluginProcessor.h`):
1. `EmptyAudioAndMidiRegions`: No audio loaded
2. `Recording`: Currently recording audio input
3. `Processing`: Running transcription (background thread)
4. `PopulatedAudioAndMidiRegions`: Transcription complete, ready for playback/export

State is stored in:
- `AudioProcessorValueTreeState` (APVTS): Automatable parameters (sensitivity, quantization, etc.)
- `ValueTree`: General plugin state (non-automatable settings)

### Threading Model

- Audio thread: Handles `processBlock()` for audio recording/playback
- Background thread pool: Runs transcription jobs via `TranscriptionManager`
- Timer callbacks: Check for job completion and update UI (see `TranscriptionManager::timerCallback()`)

### Model Details

BasicPitch is split into two parts:
1. **Features** (ONNXRuntime): CQT + Harmonic Stacking → `features_model.ort`
2. **CNN** (RTNeural): 4 sequential 2D conv models → `.json` weight files in `Lib/ModelData/`

The CNN was split into 4 models to work around RTNeural's sequential processing requirement. Model weights are embedded as binary data via `juce_add_binary_data()`.

## Reusing the Transcription Engine

The core transcription code in `Lib/Model/` and model weights in `Lib/ModelData/` are designed to be reusable in other projects. Key files:
- `BasicPitch.h/.cpp`: Main API
- `Features.h/.cpp`: CQT feature extraction
- `BasicPitchCNN.h/.cpp`: CNN inference (built separately with optimization flags)
- `Notes.h/.cpp`: Posteriorgram → MIDI conversion

## Special Considerations

### BasicPitchCNN Compilation

`BasicPitchCNN.cpp` is compiled as a separate static library with forced optimization (`-O3` on GCC/Clang, `/O2` on MSVC) even in Debug builds to maintain real-time performance. This is controlled by the `RTNeural_Release` CMake option.

### ONNXRuntime Dependencies

The build scripts download a custom pre-built ONNXRuntime static library from [libonnxruntime-neuralnote](https://github.com/tiborvass/libonnxruntime-neuralnote) built with [ort-builder](https://github.com/olilarkin/ort-builder). This includes a runtime-optimized ONNX model (`features_model.ort`).

On Windows with non-MSVC 19.35.x compilers, you must manually rebuild ONNXRuntime (see README.md).

### Platform-Specific Notes

- **macOS**: Code is signed and notarized for distribution. Use `sign_and_package_neuralnote_macos.sh` for release builds.
- **Windows**: Code is not signed. Users may need to bypass SmartScreen warnings.
- **Linux**: Raw binaries provided (no installer). Tests are disabled by default.

### RTNeural 2D Convolution

This project contributed 2D convolution support to RTNeural ([PR #89](https://github.com/jatinchowdhury18/RTNeural/pull/89)), which is used for the BasicPitch CNN layers.

## Packaging for Release

See `PACKAGING.md` for detailed packaging instructions:
- macOS: Use Packages.app and `sign_and_package_neuralnote_macos.sh`
- Windows: Use Inno Setup to build installer from `Installers/Windows/neuralnote.iss`

Build with:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUniversalBinary=ON -DLTO=ON
cmake --build build -j $(nproc)
```

## UI Development Patterns & Common Gotchas

### Image Handling

**Background Image Requirements:**
- All background images MUST be rescaled to **1000x640** pixels
- Use `Graphics::ResamplingQuality::highResamplingQuality` for best quality
- Both loading and reloading must include rescaling

**Example (from NeuralNoteMainView.cpp:232-233):**
```cpp
mBackgroundImage = ImageCache::getFromMemory(BinaryData::background_png, BinaryData::background_pngSize)
                       .rescaled(1000, 640, Graphics::ResamplingQuality::highResamplingQuality);
```

**Common Mistake:** Loading/reloading images without the `.rescaled()` step causes distorted backgrounds.

### UI Colors

**Color Definitions:** `Lib/Components/UIDefines.h:66-74`
- `BLACK`: RGB(14, 14, 14)
- `WHITE_SOLID`: RGB(255, 253, 246) - off-white cream
- `WHITE_TRANSPARENT`: RGB(255, 253, 246, 0.7f)
- `KNOB_GREY`: RGB(218, 221, 217)
- Maintain color consistency across the UI

### Settings Menu Pattern

Adding items to settings menu (NeuralNoteMainView.cpp:132+):
1. Increment `item_id`
2. Create `PopupMenu::Item` with label
3. Set ID and enabled state
4. For toggle items: add to `mSettingsMenuItemsShouldBeTicked`
5. Define action lambda
6. Call `setAction()` and `addItem()`

### Keyboard Shortcuts

Handled in `NeuralNoteMainView::keyPressed()`:
- Use `ModifierKeys::commandModifier` for Cmd/Ctrl
- Return `true` if key handled, `false` otherwise
- Check `KeyPress` equality: `key == KeyPress('b', ModifierKeys::commandModifier, 0)`

### Hot-Reload Background Feature

**Location:** Settings menu → "Reload Background" or `Cmd+B`
- Checks `~/Desktop/background.png` first
- Falls back to embedded `BinaryData::background_png`
- Always rescales to 1000x640
- Useful for testing different background designs without rebuilding

## Naming Conventions

- Never name things "FIXED" or "UPDATED" in filenames or app names
- Use descriptive, neutral names for builds and artifacts