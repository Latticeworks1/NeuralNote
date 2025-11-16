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
- Git workflow: working in cloned repo, files currently untracked
- Need to maintain THOUGHTS.md as ongoing development log

### Misconceptions and Corrections

**IMPORTANT**: Tracking errors in reasoning to improve future analysis

1. **Initial misconception about codebase location**:
   - **Wrong assumption**: When user said "analyze this codebase" while in `/Applications/NeuralNote.app/`, I initially treated that as the target directory
   - **What I found**: Only found compiled binary (Mach-O universal binary), Info.plist, and resources - no source code
   - **Correction**: Realized this was just a macOS application bundle (the compiled/installed app), not the source repository
   - **Action taken**: Asked user for source location, they provided GitHub URL, cloned to `/tmp/NeuralNote/`
   - **Lesson**: Always verify directory contains source code before assuming it's a development repository. `.app` directories on macOS are compiled bundles, not source code.
