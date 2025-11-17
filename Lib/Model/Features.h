//
// Created by Damien Ronssin on 04.03.23.
//

#ifndef Features_h
#define Features_h

#include "cassert"
#include <onnxruntime_cxx_api.h>

#include "BinaryData.h"
#include "BasicPitchConstants.h"

/**
 * Class to compute the CQT and harmonically stack those. Output of this can be given as input to Basic Pitch cnn.
 */
class Features
{
public:
    Features();

    ~Features() = default;

    /**
     * Check if ONNX model was successfully loaded
     * @return true if model initialized successfully, false otherwise
     */
    bool isInitialized() const { return mIsInitialized; }

    /**
     * Get error message if initialization failed
     * @return Error message string, or empty string if no error
     */
    const std::string& getErrorMessage() const { return mErrorMessage; }

    /**
     * Compute features for full audio signal
     * @param inAudio Input audio. Should contain inNumSamples
     * @param inNumSamples Number of samples in inAudio
     * @param outNumFrames Number of frames that have been computed.
     * @return Pointer to features, or nullptr if not initialized.
     */
    const float* computeFeatures(float* inAudio, size_t inNumSamples, size_t& outNumFrames);

private:
    // ONNX Runtime Data
    std::vector<Ort::Value> mInput;
    std::vector<Ort::Value> mOutput;

    std::array<int64_t, 3> mInputShape;

    // Input and output names of model
    const char* mInputNames[1] = {"input_1"};
    const char* mOutputNames[1] = {"harmonic_stacking"};

    // ONNX Runtime
    Ort::MemoryInfo mMemoryInfo;
    Ort::SessionOptions mSessionOptions;
    Ort::Env mEnv;
    Ort::Session mSession;
    Ort::RunOptions mRunOptions;

    // Initialization state
    bool mIsInitialized = false;
    std::string mErrorMessage;
};

#endif // Features_h
