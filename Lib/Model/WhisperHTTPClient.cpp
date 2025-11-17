#include "WhisperHTTPClient.h"

WhisperHTTPClient::WhisperHTTPClient(const juce::String& serviceUrl)
    : mServiceUrl(serviceUrl)
{
    if (mServiceUrl.endsWithChar('/')) {
        mServiceUrl = mServiceUrl.dropLastCharacters(1);
    }
}

bool WhisperHTTPClient::isServiceAvailable()
{
    return sendHealthCheck();
}

bool WhisperHTTPClient::sendHealthCheck()
{
    try {
        juce::URL healthUrl(mServiceUrl + "/health");

        auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                          .withConnectionTimeoutMs(5000);

        std::unique_ptr<juce::InputStream> stream(healthUrl.createInputStream(options));

        if (stream == nullptr) {
            mLastError = "Failed to connect to Whisper service at " + mServiceUrl;
            return false;
        }

        juce::String response = stream->readEntireStreamAsString();

        auto json = juce::JSON::parse(response);
        if (!json.isObject()) {
            mLastError = "Invalid response from service";
            return false;
        }

        juce::String status = json.getProperty("status", "").toString();
        if (status != "healthy") {
            mLastError = "Service is not healthy: " + json.getProperty("message", "Unknown error").toString();
            return false;
        }

        mLastError = "";
        return true;

    } catch (const std::exception& e) {
        mLastError = juce::String("Health check failed: ") + e.what();
        return false;
    }
}

bool WhisperHTTPClient::sendTranscriptionRequest(const juce::var& requestBody, juce::var& response)
{
    try {
        juce::URL transcribeUrl(mServiceUrl + "/transcribe");

        juce::String jsonRequest = juce::JSON::toString(requestBody, false);

        auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                          .withConnectionTimeoutMs(mTimeoutMs)
                          .withExtraHeaders("Content-Type: application/json");

        juce::MemoryBlock postData(jsonRequest.toRawUTF8(), jsonRequest.getNumBytesAsUTF8());

        std::unique_ptr<juce::InputStream> stream(
            transcribeUrl.withPOSTData(postData).createInputStream(options)
        );

        if (stream == nullptr) {
            mLastError = "Failed to send transcription request";
            return false;
        }

        juce::String responseText = stream->readEntireStreamAsString();

        response = juce::JSON::parse(responseText);
        if (!response.isObject()) {
            mLastError = "Invalid JSON response from service";
            return false;
        }

        if (response.hasProperty("error")) {
            mLastError = response.getProperty("error", "Unknown error").toString();
            return false;
        }

        mLastError = "";
        return true;

    } catch (const std::exception& e) {
        mLastError = juce::String("Transcription request failed: ") + e.what();
        return false;
    }
}

bool WhisperHTTPClient::transcribe(const float* audioData,
                                   int numSamples,
                                   const juce::String& language,
                                   std::vector<TimedWord>& outWords)
{
    outWords.clear();

    if (audioData == nullptr || numSamples == 0) {
        mLastError = "Invalid audio data";
        return false;
    }

    // Build request JSON
    juce::var requestBody(new juce::DynamicObject());

    // Convert audio samples to JSON array
    juce::Array<juce::var> audioArray;
    audioArray.ensureStorageAllocated(numSamples);
    for (int i = 0; i < numSamples; ++i) {
        audioArray.add(audioData[i]);
    }
    requestBody.getDynamicObject()->setProperty("audio", audioArray);
    requestBody.getDynamicObject()->setProperty("sample_rate", 16000);

    if (language.isNotEmpty()) {
        requestBody.getDynamicObject()->setProperty("language", language);
    }

    requestBody.getDynamicObject()->setProperty("task", "transcribe");

    // Send request
    juce::var response;
    if (!sendTranscriptionRequest(requestBody, response)) {
        return false;
    }

    // Parse response
    if (!response.hasProperty("words")) {
        mLastError = "Response missing 'words' field";
        return false;
    }

    const juce::Array<juce::var>* wordsArray = response.getProperty("words", juce::var()).getArray();
    if (wordsArray == nullptr) {
        mLastError = "'words' field is not an array";
        return false;
    }

    // Convert to TimedWord structs
    for (const auto& wordVar : *wordsArray) {
        if (!wordVar.isObject()) {
            continue;
        }

        TimedWord word;
        word.text = wordVar.getProperty("text", "").toString().toStdString();
        word.startTime = static_cast<double>(wordVar.getProperty("start", 0.0));
        word.endTime = static_cast<double>(wordVar.getProperty("end", 0.0));
        word.confidence = static_cast<float>(wordVar.getProperty("confidence", 1.0));

        outWords.push_back(word);
    }

    mLastError = "";
    return true;
}

juce::var WhisperHTTPClient::getModelInfo()
{
    try {
        juce::URL infoUrl(mServiceUrl + "/info");

        auto options = juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inAddress)
                          .withConnectionTimeoutMs(5000);

        std::unique_ptr<juce::InputStream> stream(infoUrl.createInputStream(options));

        if (stream == nullptr) {
            return juce::var();
        }

        juce::String response = stream->readEntireStreamAsString();
        return juce::JSON::parse(response);

    } catch (...) {
        return juce::var();
    }
}
