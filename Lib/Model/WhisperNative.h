#pragma once

#include "WhisperConstants.h"
#include <vector>
#include <string>
#include <memory>
#include <sstream>

// Forward declarations for whisper.cpp types
struct whisper_context;
struct whisper_full_params;

/**
 * Native C++ Whisper implementation using whisper.cpp
 * Fully self-contained, no external services required
 */
class WhisperNative
{
public:
    WhisperNative();
    ~WhisperNative();

    /**
     * Load model from file
     * @param modelPath Path to .bin model file
     * @return true if successful
     */
    bool loadModel(const std::string& modelPath);

    /**
     * Check if model is loaded and ready
     */
    bool isInitialized() const { return mContext != nullptr; }

    /**
     * Get error message if initialization failed
     */
    const std::string& getErrorMessage() const { return mErrorMessage; }

    /**
     * Transcribe audio to text with word-level timestamps
     * @param audioData 16kHz mono float32 audio
     * @param numSamples Number of samples
     * @param language Language code (e.g., "en", "auto" for detection)
     * @param outWords Output vector of timed words
     * @return true if successful
     */
    bool transcribe(const float* audioData,
                   int numSamples,
                   const std::string& language,
                   std::vector<TimedWord>& outWords);

    /**
     * Get full transcription text
     */
    std::string getFullText() const;

    /**
     * Clear previous results
     */
    void reset();

private:
    whisper_context* mContext = nullptr;
    std::vector<TimedWord> mTimedWords;
    std::string mErrorMessage;
    std::string mFullText;

    // Model search paths
    std::vector<std::string> getModelSearchPaths() const;
    std::string findModel(const std::string& modelName) const;
};
