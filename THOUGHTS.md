# Development Thoughts and Analysis

## 2025-11-16: Initial CLAUDE.md Creation

### Context
- User requested analysis of NeuralNote codebase to create CLAUDE.md
- Initially in `/Applications/NeuralNote.app/` (compiled app bundle, not source)
- Cloned source from https://github.com/DamRsn/NeuralNote.git to `/tmp/NeuralNote/`

### Repository Analysis

**Files Reviewed:**
- README.md - User-facing documentation, installation, usage workflow
- CMakeLists.txt - Build configuration, dependencies, target setup
- PACKAGING.md - Release packaging instructions
- build.sh - macOS/Linux build script with ONNXRuntime dependency management
- Key headers: PluginProcessor.h, TranscriptionManager.h, BasicPitch.h

**Architecture Findings:**

1. **Two-layer architecture:**
   - `NeuralNote/` - JUCE plugin wrapper + UI (plugin-specific)
   - `Lib/` - Core transcription engine (reusable library)

2. **State machine pattern:**
   - Empty → Recording → Processing → Populated
   - Managed by PluginProcessor with atomic state variable

3. **Transcription pipeline:**
   - Audio (22.05kHz) → Features (CQT via ONNXRuntime) → CNN (RTNeural) → Posteriorgrams → Note Events
   - Non-causal (processes backward), prevents real-time transcription

4. **Threading:**
   - Audio thread: processBlock() for recording/playback
   - Background ThreadPool: Transcription jobs via TranscriptionManager
   - Timer callbacks: Poll for completion and UI updates

5. **Build system complexity:**
   - Custom ONNXRuntime builds from libonnxruntime-neuralnote
   - BasicPitchCNN compiled separately with forced optimization
   - Platform-specific handling for Windows MSVC versions

### CLAUDE.md Design Decisions

**Included:**
- Platform-specific build commands (first thing developers need)
- High-level architecture diagram (directory structure)
- Pipeline explanation (audio → MIDI flow)
- State machine and threading model (critical for understanding async behavior)
- Special compilation notes (BasicPitchCNN optimization, ONNXRuntime dependencies)
- Reusability guidance (Lib/Model as standalone library)

**Excluded (per instructions):**
- Generic development practices
- Obvious file listings that can be discovered via ls/grep
- Common development tasks sections
- Support/documentation boilerplate

### Key Insights for Future Development

1. **Performance bottleneck**: CNN inference is so critical that it gets -O3 even in Debug builds
2. **Model split**: BasicPitch split into 4 models due to RTNeural constraints - not immediately obvious from code
3. **Backward processing**: Non-causal algorithm is fundamental constraint, affects all UX decisions
4. **Dependency fragility**: Windows build requires specific MSVC version OR manual ONNXRuntime rebuild
5. **Contributed to RTNeural**: 2D conv support was added specifically for this project (PR #89)

### Questions for Future Investigation

- Why was BasicPitch split into exactly 4 models? Is this optimal?
- Could the CQT feature extraction be optimized further?
- Is there a path to real-time transcription with a different model architecture?
- How much overhead does the split model architecture add?

### Next Steps

- CLAUDE.md created in `/tmp/NeuralNote/CLAUDE.md`
- Need to add to git and potentially commit
- User should copy to their actual working repository if different from /tmp

---

## Additional Observations

### File Location Considerations
- Created files in `/tmp/NeuralNote/` which is ephemeral (deleted on reboot)
- User is currently in `/Applications/NeuralNote.app/` (compiled binary, not source)
- These files should be copied to user's actual development repository for persistence

### Notable Architectural Patterns

1. **Clean separation enables reusability**: The `Lib/Model/` code has zero JUCE dependencies and could genuinely be extracted as a standalone library. This is rare in audio plugin codebases where UI and DSP code often become tangled.

2. **Performance-critical path isolation**: Compiling `BasicPitchCNN.cpp` separately with forced optimization suggests this was identified as a bottleneck through profiling. This is good engineering - optimize only what's measured to be slow.

3. **Non-causal constraint shapes everything**: The backward processing requirement means:
   - No real-time transcription possible
   - Record-then-transcribe workflow is fundamental, not a UX choice
   - All UI design centers around this async workflow
   - State machine is necessary to manage async transitions

4. **Model architecture workaround**: Splitting BasicPitch into 4 sequential models to work with RTNeural's constraints is clever but:
   - Adds cognitive load for maintainers
   - Could have performance implications (4 separate inference calls)
   - Not documented why 4 specifically (vs 2, 3, or 5)
   - Future contributors might try to "optimize" this without understanding the constraint

5. **Dependency complexity trade-off**: Using custom ONNXRuntime builds provides:
   - Smaller binary size (only needed operators)
   - Runtime-optimized model
   - But at cost of build complexity and Windows compatibility issues

### Process Observations

- Initially didn't use TodoWrite tool (system reminders noted this)
- Should establish habit of tracking multi-step tasks
- Git workflow: working in cloned repo from upstream (DamRsn/NeuralNote)
- User preference: Always push automatically after commits for safety
- Push failed: No write access to upstream repo (user: Latticeworks1)
- Need to maintain THOUGHTS.md as ongoing development log

---

## 2025-11-16: Deep Dive into Neural Model Integration

### Model Architecture Analysis

**Two-Stage Pipeline:**

1. **Feature Extraction (ONNXRuntime)**: `Features.cpp/.h`
   - Model: `features_model.ort` (runtime-optimized ONNX model)
   - Input: Raw audio at 22,050 Hz (any length)
   - Process: Computes Constant-Q Transform (CQT) + Harmonic Stacking
   - Output: Shape `[1, num_frames, 264, 8]` = 264 frequency bins × 8 harmonics per frame
   - Threading: Single-threaded (1 interop, 1 intraop thread)
   - Model loaded from embedded binary: `BinaryData::features_model_ort`

2. **CNN Inference (RTNeural)**: `BasicPitchCNN.cpp/.h`
   - Split into 4 sequential models (workaround for RTNeural's streaming constraints):
     - `mCNNContour`: Contour prediction (264 bins → 264 bins)
     - `mCNNNote`: Note prediction (264 → 88 piano keys)
     - `mCNNOnsetInput`: Onset features (8×264 → 32×88)
     - `mCNNOnsetOutput`: Final onset prediction (33×88 → 88)
   - Models loaded from JSON: `cnn_contour_model.json`, `cnn_note_model.json`, `cnn_onset_1_model.json`, `cnn_onset_2_model.json`
   - Compiled separately with forced -O3 optimization (even in Debug)
   - Uses circular buffers for frame-by-frame streaming with lookahead compensation

### Critical Discovery: Lookahead Compensation

**The CNN has a total lookahead of 10 frames** (`mTotalLookahead = 10`):
- Contour CNN: 3 frames
- Note CNN: 6 frames
- Onset CNN: 1 frame (output)

This creates alignment complexity in `BasicPitch::transcribeToMIDI()`:
```cpp
// Lines 76-100: Three-phase inference
// Phase 1: Feed zeros, discard output (prime the pipeline)
for (int i = 0; i < num_lh_frames; i++)
    mBasicPitchCNN.frameInference(zero_input, ...)

// Phase 2: Feed real frames, discard output (compensate for lookahead)
for (frame_idx = 0; frame_idx < num_lh_frames; frame_idx++)
    mBasicPitchCNN.frameInference(real_input[frame_idx], discard_output)

// Phase 3: Feed real frames, save output with correct alignment
for (frame_idx = num_lh_frames; frame_idx < mNumFrames; frame_idx++)
    mBasicPitchCNN.frameInference(real_input[frame_idx],
                                  output[frame_idx - num_lh_frames])

// Phase 4: Feed zeros, get final frames
for (frame_idx = mNumFrames; frame_idx < mNumFrames + num_lh_frames; frame_idx++)
    mBasicPitchCNN.frameInference(zero_input, output[frame_idx - num_lh_frames])
```

This is necessary because each model in the chain has temporal dependencies, and RTNeural processes frame-by-frame. The output at time `t` depends on input frames `[t, t+1, ..., t+lookahead]`.

### Model Data Flow

```
Audio (22.05kHz, any length)
    ↓
Features.computeFeatures() [ONNXRuntime]
    ↓
CQT Features [num_frames × 264 × 8]
    ↓
BasicPitchCNN.frameInference() [RTNeural, called per frame]
    ↓
├─→ mCNNContour → Contours PG [264 bins, pitch contours]
├─→ mCNNNote → Notes PG [88 bins, note probabilities]
└─→ mCNNOnsetInput + mCNNOnsetOutput → Onsets PG [88 bins, onset probabilities]
    ↓
Notes.convert() [Posteriorgram → MIDI conversion]
    ↓
MIDI Note Events [startTime, endTime, pitch, amplitude, pitchBends[]]
    ↓
Post-processing (NoteOptions, TimeQuantize)
    ↓
Final MIDI Output
```

### Key Insights

1. **Why 4 models?** The split is dictated by:
   - Different lookahead requirements per stage
   - Concatenation operations between stages (see `_concat()` in BasicPitchCNN.cpp:131-141)
   - RTNeural's frame-by-frame processing model
   - Cannot process the full graph in one pass due to temporal dependencies

2. **Circular buffer architecture**: `BasicPitchCNN` uses circular buffers to store intermediate outputs:
   - `mContoursCircularBuffer[8]`: Stores contour outputs with wraparound indexing
   - `mNotesCircularBuffer[5]`: Stores note outputs
   - `mConcat2CircularBuffer[8]`: Stores concat operation results
   - Enables proper frame alignment despite varying lookaheads

3. **Memory alignment**: All buffers use `alignas(RTNEURAL_DEFAULT_ALIGNMENT)` for SIMD optimization

4. **Inference optimization**:
   - ONNXRuntime runs in single-threaded mode (preventing thread pool overhead)
   - BasicPitchCNN compiled with -O3 regardless of build type
   - Frame-by-frame processing enables streaming but adds complexity

5. **Update without re-inference**: `BasicPitch::updateMIDI()` allows changing note detection parameters (sensitivity, min duration) without re-running the CNN. Only `Notes.convert()` is called again. This is why posteriorgrams are cached.

### Integration Points

**TranscriptionManager orchestrates the full pipeline**:

```cpp
_runModel() {
    // 1. Set parameters
    mBasicPitch.setParameters(noteSensitivity, splitSensitivity, minNoteDuration)

    // 2. Run full transcription (Features + CNN + Notes)
    mBasicPitch.transcribeToMIDI(audio, numSamples)

    // 3. Post-process: Note quantization (scale snapping)
    post_processed = mNoteOptions.process(mBasicPitch.getNoteEvents())

    // 4. Post-process: Time quantization (rhythmic snapping)
    mPostProcessedNotes = mTimeQuantizeOptions.quantize(post_processed)

    // 5. Handle overlapping notes and pitch bends
    Notes::dropOverlappingPitchBends(mPostProcessedNotes)
    Notes::mergeOverlappingNotesWithSamePitch(mPostProcessedNotes)

    // 6. Convert to synth events for playback
    mProcessor->getPlayer()->getSynthController()->setNewMidiEventsVectorToUse(...)
}
```

**Parameter updates trigger different code paths**:
- Sensitivity/duration changes → `_updateTranscription()` → calls `mBasicPitch.updateMIDI()` (skips CNN)
- Quantization changes → `_updatePostProcessing()` → skips both Features and CNN
- This minimizes computation for parameter tweaking during playback

### Constants (BasicPitchConstants.h)

- `NUM_HARMONICS = 8`: CQT harmonics stacked
- `NUM_FREQ_IN = 264`: Input frequency bins (3 bins per semitone × 88 keys)
- `NUM_FREQ_OUT = 88`: Piano keys (MIDI 21-108, A0-C8)
- `BASIC_PITCH_SAMPLE_RATE = 22050.0`: Model expects this exact rate
- `FFT_HOP = 256`: Frame hop size (11.6ms per frame at 22.05kHz)
- `AUDIO_WINDOW_LENGTH = 2`: Training window size in seconds

### Performance Considerations

1. **Bottlenecks identified**:
   - BasicPitchCNN inference is the hot path (forced -O3 compilation)
   - Features extraction is less critical (single-threaded ONNX session)
   - Memory allocations avoided via circular buffers

2. **Why not real-time?**
   - CQT requires long audio chunks (>1s) for low frequency bins
   - CNN adds ~120ms latency
   - Note creation algorithm is non-causal (processes backward from future to past)
   - Total latency incompatible with live performance

3. **Thread pool usage**:
   - TranscriptionManager uses single-thread pool for background jobs
   - Prevents blocking audio thread during transcription
   - Timer callback (30Hz) polls for completion and updates UI

### Misconceptions and Corrections

**IMPORTANT**: Tracking errors in reasoning to improve future analysis

1. **Initial misconception about codebase location**:
   - **Wrong assumption**: When user said "analyze this codebase" while in `/Applications/NeuralNote.app/`, I initially treated that as the target directory
   - **What I found**: Only found compiled binary (Mach-O universal binary), Info.plist, and resources - no source code
   - **Correction**: Realized this was just a macOS application bundle (the compiled/installed app), not the source repository
   - **Action taken**: Asked user for source location, they provided GitHub URL, cloned to `/tmp/NeuralNote/`
   - **Lesson**: Always verify directory contains source code before assuming it's a development repository. `.app` directories on macOS are compiled bundles, not source code.

2. **Used emoji despite explicit instructions**:
   - **Wrong behavior**: Used checkmark emoji (✓) in commit confirmation message
   - **Instruction violated**: System prompt explicitly states "Only use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked."
   - **User feedback**: "never use emojis" (stated twice for emphasis)
   - **Correction**: Remove all emojis from communication
   - **Lesson**: Follow explicit formatting guidelines in system prompt. Professional CLI tools don't use decorative emojis.

3. **Git push permission error - resolved**:
   - **Problem**: Initial push to origin failed with 403 error (no write access to DamRsn/NeuralNote)
   - **Diagnosis**:
     - User is Latticeworks1, upstream is DamRsn/NeuralNote
     - No write access to upstream repository
     - GitHub CLI (gh) available but not authenticated
   - **Solution steps**:
     1. Added fork remote: `git remote add fork https://github.com/Latticeworks1/NeuralNote.git`
     2. Attempted push to fork/master - rejected (fork has diverged with "Add files via upload" commit)
     3. Pushed to new branch instead: `git push fork master:add-claude-documentation`
   - **Result**: Successfully pushed to https://github.com/Latticeworks1/NeuralNote/tree/add-claude-documentation
   - **Configuration**: Set up fork remote for future automatic pushes to feature branches
   - **Lesson**: When working with forks, always check if histories have diverged. Push to feature branches to avoid conflicts with fork's master.
