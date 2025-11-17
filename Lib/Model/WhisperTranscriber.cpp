#include "WhisperTranscriber.h"
#include <sstream>

std::vector<TimedWord> WhisperTranscriber::transcribeToText(float* inAudio, int inNumSamples)
{
    // Clear previous results
    mTimedWords.clear();

    // Check if model is initialized
    if (!mWhisperONNX.isInitialized()) {
        return mTimedWords;
    }

    // Validate input
    if (inAudio == nullptr || inNumSamples == 0) {
        return mTimedWords;
    }

    try {
        // Step 1: Compute mel-spectrogram features
        size_t numFrames = 0;
        const float* melFeatures = mWhisperONNX.computeMelSpectrogram(inAudio, inNumSamples, numFrames);

        if (melFeatures == nullptr || numFrames == 0) {
            return mTimedWords;
        }

        // Step 2: Run encoder to get audio features
        const float* encoderOutput = mWhisperONNX.runEncoder(melFeatures, numFrames);

        if (encoderOutput == nullptr) {
            return mTimedWords;
        }

        // Step 3: Run decoder to generate text tokens
        std::vector<int> tokens;
        bool success = mWhisperONNX.runDecoder(encoderOutput, mLanguage, tokens);

        if (!success || tokens.empty()) {
            return mTimedWords;
        }

        // Step 4: Convert tokens to timed words
        mTimedWords = mWhisperONNX.tokensToTimedWords(tokens);

        return mTimedWords;

    } catch (const std::exception& e) {
        // Log error and return empty result
        mTimedWords.clear();
        return mTimedWords;
    }
}

std::string WhisperTranscriber::getFullText() const
{
    if (mTimedWords.empty()) {
        return "";
    }

    std::stringstream ss;
    for (size_t i = 0; i < mTimedWords.size(); ++i) {
        ss << mTimedWords[i].text;
        if (i < mTimedWords.size() - 1) {
            ss << " ";
        }
    }

    return ss.str();
}

void WhisperTranscriber::reset()
{
    mTimedWords.clear();
}
