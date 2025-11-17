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

    // Get audio from processor (resampled to 16kHz)
    auto* sourceAudioManager = mProcessor->getSourceAudioManager();

    // TODO: Implement getAudioResampled16k() in SourceAudioManager
    // For now, this is a placeholder
    // auto* audio16k = sourceAudioManager.getAudioResampled16k();
    // int numSamples = sourceAudioManager.getNumSamples16k();

    // Placeholder: Skip actual transcription until audio resampling is implemented
    // mWhisperTranscriber.transcribeToText(audio16k, numSamples);

    DBG("Text transcription job executed (placeholder - awaiting 16kHz audio implementation)");

    // Signal UI update
    mShouldUpdateDisplay = true;
}

void TextTranscriptionManager::_updateTranscriptionDisplay()
{
    mShouldUpdateDisplay = false;

    // TODO: Update TextRegion UI component with new transcription
    // This will be implemented when UI components are added

    DBG("Text transcription display updated");
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
}

void TextTranscriptionManager::setLanguage(WhisperConstants::Language language)
{
    mWhisperTranscriber.setLanguage(language);
}

WhisperConstants::Language TextTranscriptionManager::getLanguage() const
{
    return mWhisperTranscriber.getLanguage();
}
