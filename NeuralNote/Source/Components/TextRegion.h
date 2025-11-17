#pragma once

#include <JuceHeader.h>
#include "WhisperTranscriber.h"
#include "PluginProcessor.h"
#include "UIDefines.h"

/**
 * UI component to display transcribed text with word-level timestamps
 * Shows text overlaid on the audio timeline, similar to subtitles
 */
class TextRegion : public Component, public Timer
{
public:
    TextRegion(NeuralNoteAudioProcessor* processor);

    void resized() override;

    void paint(Graphics& g) override;

    void timerCallback() override;

    void setTimedWords(const std::vector<TimedWord>& words);

    void clear();

    void setZoomLevel(double inZoomLevel);

    void setViewportOffset(double offsetSeconds);

private:
    NeuralNoteAudioProcessor* mProcessor;

    std::vector<TimedWord> mTimedWords;

    double mZoomLevel = 1.0;
    double mViewportOffset = 0.0;

    // Returns the word that should be displayed at the current playback time
    std::string getCurrentWord() const;

    // Convert time in seconds to x-position in pixels
    float timeToPixel(double timeInSeconds) const;
};
