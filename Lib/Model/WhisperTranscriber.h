#pragma once

#include "WhisperONNX.h"
#include "WhisperHTTPClient.h"
#include "WhisperConstants.h"
#include <vector>
#include <string>
#include <memory>

/**
 * Main API class for speech-to-text transcription using Whisper model
 * Provides high-level interface for converting audio to transcribed text
 *
 * Supports two backends:
 * 1. ONNX Runtime (local, embedded models)
 * 2. HTTP Service (Hugging Face Transformers via Python service)
 */
class WhisperTranscriber
{
public:
    enum class Backend {
        ONNX,          // Use local ONNX Runtime models
        HTTPService,   // Use remote HTTP service (Hugging Face Transformers)
        Auto           // Auto-select: HTTP if available, fallback to ONNX
    };

    WhisperTranscriber(Backend backend = Backend::Auto,
                       const juce::String& serviceUrl = "http://127.0.0.1:8765");
    ~WhisperTranscriber() = default;

    /**
     * Check if the Whisper backend is ready
     * @return true if backend is ready to use, false if initialization failed
     */
    bool isInitialized() const;

    /**
     * Get error message if initialization failed
     * @return Error message string, or empty string if no error
     */
    const std::string& getErrorMessage() const;

    /**
     * Get the active backend being used
     * @return Active backend type
     */
    Backend getActiveBackend() const { return mActiveBackend; }

    /**
     * Set language for transcription
     * @param language Target language (use Language::Auto for automatic detection)
     */
    void setLanguage(WhisperConstants::Language language) { mLanguage = language; }

    /**
     * Get current language setting
     * @return Current language
     */
    WhisperConstants::Language getLanguage() const { return mLanguage; }

    /**
     * Transcribe audio to text with word-level timestamps
     * @param inAudio Pointer to raw audio (must be at 16000 Hz)
     * @param inNumSamples Number of input samples
     * @return Vector of timed words with timestamps and confidence scores
     */
    std::vector<TimedWord> transcribeToText(float* inAudio, int inNumSamples);

    /**
     * Get the last transcription result
     * @return Vector of timed words from last transcription
     */
    const std::vector<TimedWord>& getTimedWords() const { return mTimedWords; }

    /**
     * Get full transcription as single string
     * @return Complete transcribed text
     */
    std::string getFullText() const;

    /**
     * Clear previous transcription results
     */
    void reset();

private:
    void selectBackend(Backend preferredBackend);

    Backend mRequestedBackend;
    Backend mActiveBackend;

    WhisperONNX mWhisperONNX;
    std::unique_ptr<WhisperHTTPClient> mHTTPClient;

    WhisperConstants::Language mLanguage = WhisperConstants::Language::Auto;
    std::vector<TimedWord> mTimedWords;
    mutable std::string mErrorMessage;
};
