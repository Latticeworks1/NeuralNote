#include "TextRegion.h"

TextRegion::TextRegion(NeuralNoteAudioProcessor* processor)
    : mProcessor(processor)
{
    startTimerHz(30); // Refresh at 30 Hz for smooth playback display
}

void TextRegion::resized()
{
    // Layout is handled by parent component
}

void TextRegion::paint(Graphics& g)
{
    // Debug: Always show we're painting
    DBG("TextRegion::paint - words count: " + juce::String(mTimedWords.size()) + ", bounds: " + getLocalBounds().toString());

    if (mTimedWords.empty()) {
        // Semi-transparent background
        g.setColour(BLACK.withAlpha(0.3f));
        g.fillRect(getLocalBounds());

        // Bright text for visibility
        g.setColour(Colours::yellow);
        g.setFont(Font(FontOptions(Font::bold)).withPointHeight(16.0f));
        g.drawText("Text transcription will appear here",
                   getLocalBounds(),
                   Justification::centred);
        return;
    }

    g.setColour(BLACK.withAlpha(0.35f));
    g.fillRect(getLocalBounds());

    // Draw all words with timestamps
    g.setFont(Font(FontOptions()).withPointHeight(12.0f));

    for (const auto& word : mTimedWords) {
        float x = timeToPixel(word.startTime);
        float width = timeToPixel(word.endTime) - x;

        // Only draw if visible in current viewport
        if (x + width < 0 || x > getWidth()) {
            continue;
        }

        // Highlight current word during playback
        bool isCurrent = false;
        auto* player = mProcessor->getPlayer();
        if (player && player->isPlaying()) {
            double playbackTime = player->getPlayheadPositionSeconds();
            isCurrent = (playbackTime >= word.startTime && playbackTime < word.endTime);
        }

        // Draw word background
        if (isCurrent) {
            g.setColour(WHITE_TRANSPARENT.withAlpha(0.3f));
            g.fillRect(x, 0.0f, width, static_cast<float>(getHeight()));
        }

        // Draw word text
        g.setColour(isCurrent ? WHITE_SOLID : WHITE_TRANSPARENT);
        Rectangle<float> textBounds(x, 0.0f, width, static_cast<float>(getHeight()));
        g.drawText(word.text, textBounds, Justification::centredLeft);
    }

    // Draw full text at bottom as subtitle
    std::string currentWord = getCurrentWord();
    if (!currentWord.empty()) {
        g.setColour(WHITE_SOLID);
        g.setFont(Font(FontOptions(Font::bold)).withPointHeight(16.0f));
        Rectangle<int> subtitleArea = getLocalBounds().removeFromBottom(30).reduced(10, 5);
        g.fillRect(subtitleArea.toFloat());

        g.setColour(BLACK);
        g.drawText(currentWord, subtitleArea, Justification::centred);
    }
}

void TextRegion::timerCallback()
{
    // Repaint to update current word highlighting during playback
    auto* player = mProcessor->getPlayer();
    if (player && player->isPlaying() && !mTimedWords.empty()) {
        repaint();
    }
}

void TextRegion::setTimedWords(const std::vector<TimedWord>& words)
{
    mTimedWords = words;
    repaint();
}

void TextRegion::clear()
{
    mTimedWords.clear();
    repaint();
}

void TextRegion::setZoomLevel(double inZoomLevel)
{
    mZoomLevel = inZoomLevel;
    repaint();
}

void TextRegion::setViewportOffset(double offsetSeconds)
{
    mViewportOffset = offsetSeconds;
    repaint();
}

std::string TextRegion::getCurrentWord() const
{
    auto* player = mProcessor->getPlayer();
    if (!player || !player->isPlaying() || mTimedWords.empty()) {
        return "";
    }

    double playbackTime = player->getPlayheadPositionSeconds();

    for (const auto& word : mTimedWords) {
        if (playbackTime >= word.startTime && playbackTime < word.endTime) {
            return word.text;
        }
    }

    return "";
}

float TextRegion::timeToPixel(double timeInSeconds) const
{
    // Base: 100 pixels per second, adjusted by zoom
    const double basePixelsPerSecond = 100.0;
    double pixelsPerSecond = basePixelsPerSecond * mZoomLevel;

    return static_cast<float>((timeInSeconds - mViewportOffset) * pixelsPerSecond);
}
