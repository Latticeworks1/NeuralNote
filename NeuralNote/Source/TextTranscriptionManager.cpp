#include "TextTranscriptionManager.h"
#include "PluginProcessor.h"

TextTranscriptionManager::TextTranscriptionManager(NeuralNoteAudioProcessor* inProcessor)
    : mProcessor(inProcessor)
    , mThreadPool(1)
{
    // Check if Whisper model initialization succeeded
    if (!mWhisperTranscriber.isInitialized()) {
        DBG("WARNING: Whisper model not initialized: " + mWhisperTranscriber.getErrorMessage());
        // Don't show error dialog since this is a new feature and models may not be embedded yet
        // NativeMessageBox::showMessageBoxAsync(MessageBoxIconType::InfoIcon,
        //                                        "Text Transcription",
        //                                        "Whisper model not available:\n" + mWhisperTranscriber.getErrorMessage()
        //                                            + "\n\nText transcription will not be available.");
    }

    mJobLambda = [this] { _runModel(); };

    // TODO: Add parameter listeners for text transcription settings when UI is implemented
    // For example: language selection, model size, etc.

    startTimerHz(30);
}

TextTranscriptionManager::~TextTranscriptionManager()
{
    stopTimer();
}

void TextTranscriptionManager::timerCallback()
{
    if (mShouldRunNewTranscription) {
        launchTranscribeJob();
    } else if (mShouldUpdateDisplay) {
        _updateTranscriptionDisplay();
    }
}

void TextTranscriptionManager::setLaunchNewTranscription()
{
    mShouldRunNewTranscription = true;
    mShouldUpdateDisplay = false;
}

void TextTranscriptionManager::launchTranscribeJob()
{
    mShouldRunNewTranscription = false;

    if (!mWhisperTranscriber.isInitialized()) {
        DBG("Cannot launch text transcription - Whisper model not initialized");
        return;
    }

    // Launch job on background thread
    mThreadPool.addJob(mJobLambda);
}

void TextTranscriptionManager::_runModel()
{
    if (!mWhisperTranscriber.isInitialized()) {
        return;
    }

    auto* sourceAudioManager = mProcessor->getSourceAudioManager();
    if (sourceAudioManager == nullptr) {
        DBG("Text transcription skipped - missing SourceAudioManager");
        return;
    }

    float* audio16k = sourceAudioManager->getAudioResampled16k();
    const int numSamples = sourceAudioManager->getNumSamples16k();

    if (audio16k == nullptr || numSamples == 0) {
        DBG("Text transcription skipped - 16kHz audio not available yet");
        return;
    }

    auto words = mWhisperTranscriber.transcribeToText(audio16k, numSamples);
    if (words.empty()) {
        DBG("Text transcription completed but returned no tokens.");
    }

    // Signal UI update
    mShouldUpdateDisplay = true;
}

void TextTranscriptionManager::_updateTranscriptionDisplay()
{
    mShouldUpdateDisplay = false;

    const auto& words = mWhisperTranscriber.getTimedWords();
    if (words.empty()) {
        mProcessor->clearTimedWordsOnUI();
    } else {
        mProcessor->updateTimedWordsOnUI(words);
    }
}

void TextTranscriptionManager::parameterChanged(const String& parameterID, float newValue)
{
    juce::ignoreUnused(parameterID, newValue);

    // TODO: Handle parameter changes (e.g., language selection)
    // Currently no text-specific parameters defined
}

bool TextTranscriptionManager::isJobRunningOrQueued() const
{
    return mThreadPool.getNumJobs() > 0 || mShouldRunNewTranscription;
}

const std::vector<TimedWord>& TextTranscriptionManager::getTimedWords() const
{
    return mWhisperTranscriber.getTimedWords();
}

std::string TextTranscriptionManager::getFullText() const
{
    return mWhisperTranscriber.getFullText();
}

void TextTranscriptionManager::clear()
{
    mWhisperTranscriber.reset();
    mShouldRunNewTranscription = false;
    mShouldUpdateDisplay = false;
    mProcessor->clearTimedWordsOnUI();
}

void TextTranscriptionManager::setLanguage(WhisperConstants::Language language)
{
    mWhisperTranscriber.setLanguage(language);
}

WhisperConstants::Language TextTranscriptionManager::getLanguage() const
{
    return mWhisperTranscriber.getLanguage();
}
