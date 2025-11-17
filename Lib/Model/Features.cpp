//
// Created by Damien Ronssin on 04.03.23.
//

#include "Features.h"

Features::Features()
    : mMemoryInfo(nullptr)
    , mSession(nullptr)
{
    try {
        mMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        mSessionOptions.SetInterOpNumThreads(1);
        mSessionOptions.SetIntraOpNumThreads(1);

        mSession = Ort::Session(mEnv, BinaryData::features_model_ort, BinaryData::features_model_ortSize, mSessionOptions);

        mIsInitialized = true;
    } catch (const Ort::Exception& e) {
        mIsInitialized = false;
        mErrorMessage = "ONNX Runtime error during model initialization: " + std::string(e.what());
    } catch (const std::exception& e) {
        mIsInitialized = false;
        mErrorMessage = "Error during model initialization: " + std::string(e.what());
    } catch (...) {
        mIsInitialized = false;
        mErrorMessage = "Unknown error during model initialization";
    }
}

const float* Features::computeFeatures(float* inAudio, size_t inNumSamples, size_t& outNumFrames)
{
    // Check if model was successfully initialized
    if (!mIsInitialized) {
        outNumFrames = 0;
        return nullptr;
    }

    mInputShape[0] = 1;
    mInputShape[1] = static_cast<int64_t>(inNumSamples);
    mInputShape[2] = 1;

    mInput.clear();
    mInput.push_back(
        Ort::Value::CreateTensor<float>(mMemoryInfo, inAudio, inNumSamples, mInputShape.data(), mInputShape.size()));

    mOutput = mSession.Run(mRunOptions, mInputNames, mInput.data(), 1, mOutputNames, 1);

    auto out_shape = mOutput[0].GetTensorTypeAndShapeInfo().GetShape();
    assert(out_shape[0] == 1 && out_shape[2] == NUM_FREQ_IN && out_shape[3] == NUM_HARMONICS);

    outNumFrames = static_cast<size_t>(out_shape[1]);

    mInput.clear();

    // Note: The returned pointer to mOutput[0] tensor data remains valid until the next
    // Run() call or until mOutput is modified. Clearing mInput does not invalidate it.
    return mOutput[0].GetTensorData<float>();
}
