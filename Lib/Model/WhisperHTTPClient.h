#pragma once

#include <JuceHeader.h>
#include "WhisperConstants.h"
#include "WhisperONNX.h"  // For TimedWord struct

/**
 * HTTP client for communicating with the NeuralNote Whisper Service
 *
 * This client provides a bridge between the C++ plugin and the Python-based
 * Hugging Face Transformers Whisper service running locally.
 */
class WhisperHTTPClient
{
public:
    /**
     * Constructor
     * @param serviceUrl Base URL of the Whisper service (default: http://127.0.0.1:8765)
     */
    explicit WhisperHTTPClient(const juce::String& serviceUrl = "http://127.0.0.1:8765");
    ~WhisperHTTPClient() = default;

    /**
     * Check if the service is available and healthy
     * @return true if service is running and responsive
     */
    bool isServiceAvailable();

    /**
     * Transcribe audio using the remote Whisper service
     * @param audioData Pointer to audio samples (16kHz, float32, mono)
     * @param numSamples Number of samples
     * @param language Language code (e.g., "en", "es"), or empty for auto-detect
     * @param outWords Output vector of timed words with timestamps
     * @return true if transcription succeeded
     */
    bool transcribe(const float* audioData,
                   int numSamples,
                   const juce::String& language,
                   std::vector<TimedWord>& outWords);

    /**
     * Get information about the loaded model
     * @return JSON object with model info, or empty object if request fails
     */
    juce::var getModelInfo();

    /**
     * Get the last error message
     */
    juce::String getLastError() const { return mLastError; }

    /**
     * Set custom timeout for HTTP requests (in milliseconds)
     */
    void setTimeout(int timeoutMs) { mTimeoutMs = timeoutMs; }

private:
    bool sendHealthCheck();
    bool sendTranscriptionRequest(const juce::var& requestBody, juce::var& response);

    juce::String mServiceUrl;
    juce::String mLastError;
    int mTimeoutMs = 30000; // 30 seconds default timeout

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WhisperHTTPClient)
};
