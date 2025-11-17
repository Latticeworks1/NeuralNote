#include "WhisperONNX.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

#include "WhisperModelLoader.h"

namespace
{
constexpr const char* kWhisperPlaceholderMagic = "NEURALNOTE_WHISPER_PLACEHOLDER";

bool isPlaceholderModelData(const char* data, size_t size)
{
    if (data == nullptr || size == 0) {
        return true;
    }

    const size_t markerSize = std::strlen(kWhisperPlaceholderMagic);
    if (size < markerSize) {
        return false;
    }

    const char* begin = data;
    const char* end = data + size;
    return std::search(begin, end, kWhisperPlaceholderMagic, kWhisperPlaceholderMagic + markerSize) != end;
}
} // namespace

WhisperONNX::WhisperONNX()
    : mEnv(ORT_LOGGING_LEVEL_WARNING, "WhisperONNX")
    , mMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
    , mEncoderSession(nullptr)
    , mDecoderSession(nullptr)
{
    try {

        // Configure ONNX Runtime sessions
        mEncoderSessionOptions.SetInterOpNumThreads(1);
        mEncoderSessionOptions.SetIntraOpNumThreads(1);
        mDecoderSessionOptions.SetInterOpNumThreads(1);
        mDecoderSessionOptions.SetIntraOpNumThreads(1);

        const char* embeddedEncoder = BinaryData::whisper_encoder_ort;
        const char* embeddedDecoder = BinaryData::whisper_decoder_ort;

        size_t encoderSize = static_cast<size_t>(BinaryData::whisper_encoder_ortSize);
        size_t decoderSize = static_cast<size_t>(BinaryData::whisper_decoder_ortSize);

        bool hasValidEncoder = !isPlaceholderModelData(embeddedEncoder, encoderSize);
        bool hasValidDecoder = !isPlaceholderModelData(embeddedDecoder, decoderSize);

        const char* encoderData = embeddedEncoder;
        const char* decoderData = embeddedDecoder;

        if (!hasValidEncoder || !hasValidDecoder) {
            auto loadResult = WhisperModelLoader::loadFromDefaultLocations(mExternalEncoderData, mExternalDecoderData);
            if (loadResult.success) {
                encoderData = reinterpret_cast<const char*>(mExternalEncoderData.data());
                decoderData = reinterpret_cast<const char*>(mExternalDecoderData.data());
                encoderSize = mExternalEncoderData.size();
                decoderSize = mExternalDecoderData.size();
                hasValidEncoder = hasValidDecoder = true;
                mErrorMessage.clear();
            } else {
                mIsInitialized = false;
                mErrorMessage = loadResult.message;
            }
        }

        if (hasValidEncoder && hasValidDecoder) {
            mEncoderSession = Ort::Session(mEnv, encoderData, encoderSize, mEncoderSessionOptions);
            mDecoderSession = Ort::Session(mEnv, decoderData, decoderSize, mDecoderSessionOptions);
            mIsInitialized = true;
        }

        // Initialize mel filterbank
        initializeMelFilters();

        if (!mIsInitialized && mErrorMessage.empty()) {
            mErrorMessage =
                "Whisper models not embedded. Replace Lib/ModelData/whisper_encoder.ort and "
                "Lib/ModelData/whisper_decoder.ort with real ONNX Runtime blobs (placeholder implementation).";
        } else if (!mIsInitialized) {
            mErrorMessage += " (placeholder implementation)";
        }
    } catch (const Ort::Exception& e) {
        mIsInitialized = false;
        mErrorMessage = "ONNX Runtime error during Whisper model initialization: " + std::string(e.what());
    } catch (const std::exception& e) {
        mIsInitialized = false;
        mErrorMessage = "Error during Whisper model initialization: " + std::string(e.what());
    } catch (...) {
        mIsInitialized = false;
        mErrorMessage = "Unknown error during Whisper model initialization";
    }
}

void WhisperONNX::initializeMelFilters()
{
    // Initialize mel filterbank for 80 mel bins
    // This is a simplified version - full implementation would load from binary data
    const int nMels = WhisperConstants::WHISPER_N_MELS;
    const int nFft = WhisperConstants::WHISPER_N_FFT;

    mMelFilters.resize(nMels);
    for (int i = 0; i < nMels; ++i) {
        mMelFilters[i].resize(nFft / 2 + 1, 0.0f);
        // TODO: Load actual mel filter coefficients from binary data
    }
}

const float* WhisperONNX::computeMelSpectrogram(float* inAudio, size_t inNumSamples, size_t& outNumFrames)
{
    if (!mIsInitialized || inAudio == nullptr || inNumSamples == 0) {
        outNumFrames = 0;
        return nullptr;
    }

    const int hopLength = WhisperConstants::WHISPER_HOP_LENGTH;
    const int nFft = WhisperConstants::WHISPER_N_FFT;
    const int nMels = WhisperConstants::WHISPER_N_MELS;

    // Calculate number of frames
    outNumFrames = (inNumSamples - nFft) / hopLength + 1;

    // Allocate mel buffer
    mMelBuffer.resize(nMels * outNumFrames);

    // Compute STFT and apply mel filterbank
    std::vector<float> fftOutput;
    for (size_t i = 0; i < outNumFrames; ++i) {
        size_t offset = i * hopLength;
        computeFFT(inAudio + offset, nFft, fftOutput);

        // Apply mel filters
        for (int mel = 0; mel < nMels; ++mel) {
            float melValue = 0.0f;
            for (size_t freq = 0; freq < fftOutput.size(); ++freq) {
                melValue += fftOutput[freq] * mMelFilters[mel][freq];
            }
            mMelBuffer[mel * outNumFrames + i] = std::log(std::max(melValue, 1e-10f));
        }
    }

    return mMelBuffer.data();
}

void WhisperONNX::computeFFT(const float* audio, size_t numSamples, std::vector<float>& fftOutput)
{
    // Simplified FFT placeholder - would use optimized library (e.g., FFTW or vDSP)
    // For now, just return zeros
    fftOutput.resize(numSamples / 2 + 1, 0.0f);
    // TODO: Implement actual FFT computation
}

const float* WhisperONNX::runEncoder(const float* melFeatures, size_t numFrames)
{
    if (!mIsInitialized || melFeatures == nullptr || numFrames == 0) {
        return nullptr;
    }

    try {
        // Prepare input tensor shape: [1, 80, numFrames]
        std::array<int64_t, 3> inputShape = {1, WhisperConstants::WHISPER_N_MELS, static_cast<int64_t>(numFrames)};

        mEncoderInput.clear();
        mEncoderInput.push_back(Ort::Value::CreateTensor<float>(
            mMemoryInfo,
            const_cast<float*>(melFeatures),
            WhisperConstants::WHISPER_N_MELS * numFrames,
            inputShape.data(),
            inputShape.size()
        ));

        // Run encoder
        mEncoderOutput = mEncoderSession.Run(
            mRunOptions,
            mEncoderInputNames,
            mEncoderInput.data(),
            1,
            mEncoderOutputNames,
            1
        );

        // Cache encoder output
        auto outputShape = mEncoderOutput[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t outputSize = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());

        const float* outputData = mEncoderOutput[0].GetTensorData<float>();
        mEncoderOutputBuffer.assign(outputData, outputData + outputSize);

        mEncoderInput.clear();

        return mEncoderOutputBuffer.data();

    } catch (const Ort::Exception& e) {
        mErrorMessage = "ONNX Runtime error in encoder: " + std::string(e.what());
        return nullptr;
    }
}

bool WhisperONNX::runDecoder(const float* encoderOutput, WhisperConstants::Language language, std::vector<int>& tokens)
{
    if (!mIsInitialized || encoderOutput == nullptr) {
        return false;
    }

    try {
        // Initialize decoder with start token
        tokens.clear();
        tokens.push_back(WhisperConstants::TOKEN_SOT);

        // Add language token based on selected language
        const char* langCode = WhisperConstants::languageToString(language);
        // TODO: Map language code to actual token ID

        // Autoregressive decoding loop
        const int maxTokens = 448;
        for (int i = 0; i < maxTokens; ++i) {
            // Prepare decoder inputs
            // Input 1: tokens so far
            // Input 2: encoder output (audio features)
            // Input 3: offset/position

            // TODO: Implement actual decoder forward pass
            // This is a placeholder - real implementation would:
            // 1. Create token tensor from current tokens
            // 2. Pass encoder output and tokens to decoder
            // 3. Get logits for next token
            // 4. Sample/beam search for best token
            // 5. Append to tokens
            // 6. Break if EOT token generated

            // For now, just add EOT and break
            tokens.push_back(WhisperConstants::TOKEN_EOT);
            break;
        }

        return true;

    } catch (const Ort::Exception& e) {
        mErrorMessage = "ONNX Runtime error in decoder: " + std::string(e.what());
        return false;
    }
}

std::vector<TimedWord> WhisperONNX::tokensToTimedWords(const std::vector<int>& tokens)
{
    std::vector<TimedWord> result;

    // TODO: Implement token-to-text conversion
    // This would:
    // 1. Load vocabulary/tokenizer
    // 2. Decode token IDs to text
    // 3. Extract timestamp tokens
    // 4. Build TimedWord structs with start/end times

    // Placeholder: return empty result with explanation
    if (!tokens.empty()) {
        TimedWord placeholder;
        placeholder.text = "[Transcription placeholder - tokenizer not yet implemented]";
        placeholder.startTime = 0.0;
        placeholder.endTime = 0.0;
        placeholder.confidence = 0.0f;
        result.push_back(placeholder);
    }

    return result;
}
