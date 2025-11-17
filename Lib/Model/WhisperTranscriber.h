#pragma once

#include "WhisperONNX.h"
#include "WhisperConstants.h"
#include <vector>
#include <string>

/**
 * Main API class for speech-to-text transcription using Whisper model
 * Provides high-level interface for converting audio to transcribed text
 */
class WhisperTranscriber
{
public:
    WhisperTranscriber() = default;
    ~WhisperTranscriber() = default;

    /**
     * Check if the Whisper model was successfully initialized
     * @return true if model is ready to use, false if initialization failed
     */
    bool isInitialized() const { return mWhisperONNX.isInitialized(); }

    /**
     * Get error message if initialization failed
     * @return Error message string, or empty string if no error
     */
    const std::string& getErrorMessage() const { return mWhisperONNX.getErrorMessage(); }

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
    WhisperONNX mWhisperONNX;
    WhisperConstants::Language mLanguage = WhisperConstants::Language::Auto;
    std::vector<TimedWord> mTimedWords;
};
