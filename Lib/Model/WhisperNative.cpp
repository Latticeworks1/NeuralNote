#include "WhisperNative.h"
#include "whisper.h"
#include <JuceHeader.h>

WhisperNative::WhisperNative()
{
    // Try to auto-load a model from standard locations
    std::vector<std::string> modelNames = {
        "ggml-tiny.en.bin",
        "ggml-base.en.bin",
        "ggml-small.en.bin",
        "ggml-tiny.bin",
        "ggml-base.bin"
    };

    for (const auto& name : modelNames) {
        std::string path = findModel(name);
        if (!path.empty() && loadModel(path)) {
            DBG("WhisperNative: Auto-loaded model: " + juce::String(path));
            return;
        }
    }

    mErrorMessage = "No Whisper model found. Place a .bin model in Lib/ModelData/ or ~/Library/Application Support/NeuralNote/Models/";
}

WhisperNative::~WhisperNative()
{
    if (mContext) {
        whisper_free(mContext);
        mContext = nullptr;
    }
}

std::vector<std::string> WhisperNative::getModelSearchPaths() const
{
    std::vector<std::string> paths;

    // 1. Lib/ModelData (for embedded/local models)
    juce::File modelDataDir = juce::File::getCurrentWorkingDirectory()
                                  .getChildFile("Lib")
                                  .getChildFile("ModelData");
    if (modelDataDir.exists()) {
        paths.push_back(modelDataDir.getFullPathName().toStdString());
    }

    // 2. Application Support directory
#if JUCE_MAC
    juce::File appSupport = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                               .getChildFile("NeuralNote")
                               .getChildFile("Models");
#elif JUCE_WINDOWS
    juce::File appSupport = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                               .getChildFile("NeuralNote")
                               .getChildFile("Models");
#else
    juce::File appSupport = juce::File::getSpecialLocation(juce::File::userHomeDirectory)
                               .getChildFile(".local")
                               .getChildFile("share")
                               .getChildFile("NeuralNote")
                               .getChildFile("Models");
#endif

    if (appSupport.exists()) {
        paths.push_back(appSupport.getFullPathName().toStdString());
    }

    // 3. Environment variable
    if (const char* envPath = std::getenv("NEURALNOTE_WHISPER_DIR")) {
        paths.push_back(envPath);
    }

    return paths;
}

std::string WhisperNative::findModel(const std::string& modelName) const
{
    for (const auto& searchPath : getModelSearchPaths()) {
        juce::File modelFile = juce::File(searchPath).getChildFile(modelName);
        if (modelFile.existsAsFile()) {
            return modelFile.getFullPathName().toStdString();
        }
    }
    return "";
}

bool WhisperNative::loadModel(const std::string& modelPath)
{
    if (mContext) {
        whisper_free(mContext);
        mContext = nullptr;
    }

    whisper_context_params context_params = whisper_context_default_params();
    mContext = whisper_init_from_file_with_params(modelPath.c_str(), context_params);

    if (!mContext) {
        mErrorMessage = "Failed to load model from: " + modelPath;
        return false;
    }

    mErrorMessage.clear();
    DBG("WhisperNative: Loaded model from " + juce::String(modelPath));
    return true;
}

bool WhisperNative::transcribe(const float* audioData,
                               int numSamples,
                               const std::string& language,
                               std::vector<TimedWord>& outWords)
{
    outWords.clear();
    mTimedWords.clear();
    mFullText.clear();

    if (!mContext) {
        mErrorMessage = "Model not initialized";
        return false;
    }

    if (!audioData || numSamples == 0) {
        mErrorMessage = "Invalid audio data";
        return false;
    }

    // Set up parameters
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // Enable word-level timestamps
    params.token_timestamps = true;
    params.max_tokens = 0;  // No limit
    params.translate = false;

    // Set language if specified
    if (!language.empty() && language != "auto") {
        params.language = language.c_str();
    } else {
        params.language = "auto";
    }

    // Run transcription
    int result = whisper_full(mContext, params, audioData, numSamples);

    if (result != 0) {
        mErrorMessage = "Transcription failed with code: " + std::to_string(result);
        return false;
    }

    // Extract segments and words
    const int n_segments = whisper_full_n_segments(mContext);

    for (int i = 0; i < n_segments; ++i) {
        const int64_t t0 = whisper_full_get_segment_t0(mContext, i);
        const int64_t t1 = whisper_full_get_segment_t1(mContext, i);
        const char* text = whisper_full_get_segment_text(mContext, i);

        if (!text) continue;

        // Convert centiseconds to seconds
        double startTime = static_cast<double>(t0) / 100.0;
        double endTime = static_cast<double>(t1) / 100.0;

        // Split segment into words (simple whitespace split)
        std::string segmentText(text);
        std::istringstream iss(segmentText);
        std::string word;
        double wordDuration = (endTime - startTime) / std::max(1, (int)segmentText.size());

        double currentTime = startTime;
        while (iss >> word) {
            TimedWord timedWord;
            timedWord.text = word;
            timedWord.startTime = currentTime;
            timedWord.endTime = currentTime + wordDuration * word.length();
            timedWord.confidence = 1.0f;  // whisper.cpp doesn't provide confidence

            mTimedWords.push_back(timedWord);
            outWords.push_back(timedWord);

            currentTime = timedWord.endTime;
        }

        if (!mFullText.empty()) mFullText += " ";
        mFullText += segmentText;
    }

    mErrorMessage.clear();
    return true;
}

std::string WhisperNative::getFullText() const
{
    return mFullText;
}

void WhisperNative::reset()
{
    mTimedWords.clear();
    mFullText.clear();
}
