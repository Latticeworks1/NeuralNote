#pragma once

#include <JuceHeader.h>
#include "WhisperTranscriber.h"

class NeuralNoteAudioProcessor;

/**
 * Manager for text transcription using Whisper model
 * Handles background threading and coordination with UI
 */
class TextTranscriptionManager
    : public Timer
    , public AudioProcessorValueTreeState::Listener
{
public:
    explicit TextTranscriptionManager(NeuralNoteAudioProcessor* inProcessor);
    ~TextTranscriptionManager() override;

    void timerCallback() override;

    void setLaunchNewTranscription();

    void launchTranscribeJob();

    void parameterChanged(const juce::String& parameterID, float newValue) override;

    bool isJobRunningOrQueued() const;

    const std::vector<TimedWord>& getTimedWords() const;

    std::string getFullText() const;

    void clear();

    void setLanguage(WhisperConstants::Language language);

    WhisperConstants::Language getLanguage() const;

private:
    void _runModel();

    void _updateTranscriptionDisplay();

    NeuralNoteAudioProcessor* mProcessor;

    WhisperTranscriber mWhisperTranscriber;

    std::atomic<bool> mShouldRunNewTranscription = false;
    std::atomic<bool> mShouldUpdateDisplay = false;

    ThreadPool mThreadPool;
    std::function<void()> mJobLambda;
};
