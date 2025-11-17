#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <array>
#include <cstdint>

#include "BinaryData.h"
#include "WhisperConstants.h"

/**
 * Class to run Whisper ONNX models for speech-to-text transcription
 * Uses two-stage architecture: Encoder + Decoder
 */
class WhisperONNX
{
public:
    WhisperONNX();
    ~WhisperONNX() = default;

    /**
     * Check if ONNX models were successfully loaded
     * @return true if both encoder and decoder initialized successfully
     */
    bool isInitialized() const { return mIsInitialized; }

    /**
     * Get error message if initialization failed
     * @return Error message string, or empty string if no error
     */
    const std::string& getErrorMessage() const { return mErrorMessage; }

    /**
     * Compute mel-spectrogram features from raw audio
     * @param inAudio Input audio at 16kHz sample rate
     * @param inNumSamples Number of samples in inAudio
     * @param outNumFrames Number of mel frames computed
     * @return Pointer to mel-spectrogram features (80 mel bins x outNumFrames)
     */
    const float* computeMelSpectrogram(float* inAudio, size_t inNumSamples, size_t& outNumFrames);

    /**
     * Run encoder on mel-spectrogram features
     * @param melFeatures Input mel-spectrogram (80 x numFrames)
     * @param numFrames Number of time frames
     * @return Encoder hidden states (or nullptr if error)
     */
    const float* runEncoder(const float* melFeatures, size_t numFrames);

    /**
     * Run decoder with encoder output to generate text tokens
     * @param encoderOutput Hidden states from encoder
     * @param language Language to use for decoding (WhisperConstants::Language)
     * @param tokens Output vector of token IDs
     * @return true if decoding succeeded
     */
    bool runDecoder(const float* encoderOutput, WhisperConstants::Language language, std::vector<int>& tokens);

    /**
     * Decode tokens to text with timestamps
     * @param tokens Token IDs from decoder
     * @return Vector of timed words
     */
    std::vector<TimedWord> tokensToTimedWords(const std::vector<int>& tokens);

private:
    // Mel-spectrogram computation
    void initializeMelFilters();
    void computeFFT(const float* audio, size_t numSamples, std::vector<float>& fftOutput);

    // ONNX Runtime for Encoder
    Ort::MemoryInfo mMemoryInfo;
    Ort::SessionOptions mEncoderSessionOptions;
    Ort::SessionOptions mDecoderSessionOptions;
    Ort::Env mEnv;
    Ort::Session mEncoderSession;
    Ort::Session mDecoderSession;
    Ort::RunOptions mRunOptions;

    // Encoder/Decoder I/O
    std::vector<Ort::Value> mEncoderInput;
    std::vector<Ort::Value> mEncoderOutput;
    std::vector<Ort::Value> mDecoderInput;
    std::vector<Ort::Value> mDecoderOutput;

    const char* mEncoderInputNames[1] = {"mel"};
    const char* mEncoderOutputNames[1] = {"output"};
    const char* mDecoderInputNames[3] = {"tokens", "audio_features", "offset"};
    const char* mDecoderOutputNames[1] = {"logits"};

    // Mel filterbank
    std::vector<std::vector<float>> mMelFilters;
    std::vector<float> mMelBuffer;

    // State
    bool mIsInitialized = false;
    std::string mErrorMessage;

    // Encoder output cache
    std::vector<float> mEncoderOutputBuffer;

    // External model buffers when models are loaded from disk
    std::vector<uint8_t> mExternalEncoderData;
    std::vector<uint8_t> mExternalDecoderData;
};
