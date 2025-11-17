#include "WhisperTranscriber.h"
#include <sstream>

WhisperTranscriber::WhisperTranscriber(Backend backend, const juce::String& serviceUrl)
    : mRequestedBackend(backend)
    , mActiveBackend(Backend::ONNX)  // Default until selectBackend runs
{
    selectBackend(backend);

    // Create HTTP client if needed
    if (mActiveBackend == Backend::HTTPService || mRequestedBackend == Backend::Auto) {
        mHTTPClient = std::make_unique<WhisperHTTPClient>(serviceUrl);
    }
}

void WhisperTranscriber::selectBackend(Backend preferredBackend)
{
    mErrorMessage.clear();

    if (preferredBackend == Backend::HTTPService) {
        mActiveBackend = Backend::HTTPService;
        return;
    }

    if (preferredBackend == Backend::ONNX) {
        mActiveBackend = Backend::ONNX;
        return;
    }

    // Auto mode: Try HTTP service first, fallback to ONNX
    mHTTPClient = std::make_unique<WhisperHTTPClient>();
    if (mHTTPClient->isServiceAvailable()) {
        mActiveBackend = Backend::HTTPService;
        DBG("WhisperTranscriber: Using HTTP service backend");
    } else {
        if (mWhisperONNX.isInitialized()) {
            mActiveBackend = Backend::ONNX;
            DBG("WhisperTranscriber: Using ONNX Runtime backend");
        } else {
            // Neither backend available
            mActiveBackend = Backend::ONNX;  // Set to ONNX but will fail isInitialized() check
            mErrorMessage = "No Whisper backend available. HTTP service: " +
                           mHTTPClient->getLastError().toStdString() +
                           ", ONNX: " + mWhisperONNX.getErrorMessage();
            DBG("WhisperTranscriber: " + juce::String(mErrorMessage));
        }
    }
}

bool WhisperTranscriber::isInitialized() const
{
    switch (mActiveBackend) {
        case Backend::HTTPService:
            return mHTTPClient != nullptr && mHTTPClient->isServiceAvailable();
        case Backend::ONNX:
            return mWhisperONNX.isInitialized();
        case Backend::Auto:
        default:
            return false;
    }
}

const std::string& WhisperTranscriber::getErrorMessage() const
{
    if (!mErrorMessage.empty()) {
        return mErrorMessage;
    }

    switch (mActiveBackend) {
        case Backend::HTTPService:
            if (mHTTPClient) {
                // Convert juce::String to std::string
                static std::string httpError;
                httpError = mHTTPClient->getLastError().toStdString();
                return httpError;
            }
            break;
        case Backend::ONNX:
            return mWhisperONNX.getErrorMessage();
        default:
            break;
    }

    static const std::string emptyError;
    return emptyError;
}

std::vector<TimedWord> WhisperTranscriber::transcribeToText(float* inAudio, int inNumSamples)
{
    // Clear previous results
    mTimedWords.clear();
    mErrorMessage.clear();

    // Validate input
    if (inAudio == nullptr || inNumSamples == 0) {
        mErrorMessage = "Invalid audio input";
        return mTimedWords;
    }

    // Route to appropriate backend
    switch (mActiveBackend) {
        case Backend::HTTPService:
            if (mHTTPClient != nullptr) {
                juce::String languageCode;
                if (mLanguage != WhisperConstants::Language::Auto) {
                    languageCode = juce::String(WhisperConstants::languageToString(mLanguage));
                }

                bool success = mHTTPClient->transcribe(inAudio, inNumSamples, languageCode, mTimedWords);
                if (!success) {
                    mErrorMessage = mHTTPClient->getLastError().toStdString();
                }
                return mTimedWords;
            }
            break;

        case Backend::ONNX:
            if (!mWhisperONNX.isInitialized()) {
                mErrorMessage = "ONNX backend not initialized";
                return mTimedWords;
            }

            try {
                // Step 1: Compute mel-spectrogram features
                size_t numFrames = 0;
                const float* melFeatures = mWhisperONNX.computeMelSpectrogram(inAudio, inNumSamples, numFrames);

                if (melFeatures == nullptr || numFrames == 0) {
                    mErrorMessage = "Failed to compute mel-spectrogram";
                    return mTimedWords;
                }

                // Step 2: Run encoder to get audio features
                const float* encoderOutput = mWhisperONNX.runEncoder(melFeatures, numFrames);

                if (encoderOutput == nullptr) {
                    mErrorMessage = "Encoder failed";
                    return mTimedWords;
                }

                // Step 3: Run decoder to generate text tokens
                std::vector<int> tokens;
                bool success = mWhisperONNX.runDecoder(encoderOutput, mLanguage, tokens);

                if (!success || tokens.empty()) {
                    mErrorMessage = "Decoder failed";
                    return mTimedWords;
                }

                // Step 4: Convert tokens to timed words
                mTimedWords = mWhisperONNX.tokensToTimedWords(tokens);
                return mTimedWords;

            } catch (const std::exception& e) {
                mErrorMessage = std::string("ONNX transcription error: ") + e.what();
                mTimedWords.clear();
                return mTimedWords;
            }
            break;

        default:
            mErrorMessage = "Invalid backend configuration";
            break;
    }

    return mTimedWords;
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
