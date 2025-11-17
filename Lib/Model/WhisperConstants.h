#pragma once

namespace WhisperConstants {
    // Audio parameters
    static constexpr double WHISPER_SAMPLE_RATE = 16000.0;
    static constexpr int WHISPER_N_FFT = 400;
    static constexpr int WHISPER_HOP_LENGTH = 160;
    static constexpr int WHISPER_N_MELS = 80;
    static constexpr int WHISPER_CHUNK_LENGTH = 30; // seconds per chunk

    // Model parameters
    static constexpr int WHISPER_N_TEXT_CTX = 448;
    static constexpr int WHISPER_N_VOCAB = 51865;
    static constexpr int MAX_TEXT_LENGTH = 1024;

    // Special tokens
    static constexpr int TOKEN_SOT = 50258;  // Start of transcript
    static constexpr int TOKEN_EOT = 50257;  // End of transcript
    static constexpr int TOKEN_TIMESTAMP_BEGIN = 50364;

    // Language IDs (subset - can expand)
    enum class Language {
        Auto = -1,
        English = 0,
        Spanish = 1,
        French = 2,
        German = 3,
        Italian = 4,
        Portuguese = 5,
        Dutch = 6,
        Russian = 7,
        Chinese = 8,
        Japanese = 9,
        Korean = 10
    };

    static const char* languageToString(Language lang) {
        switch (lang) {
            case Language::Auto: return "auto";
            case Language::English: return "en";
            case Language::Spanish: return "es";
            case Language::French: return "fr";
            case Language::German: return "de";
            case Language::Italian: return "it";
            case Language::Portuguese: return "pt";
            case Language::Dutch: return "nl";
            case Language::Russian: return "ru";
            case Language::Chinese: return "zh";
            case Language::Japanese: return "ja";
            case Language::Korean: return "ko";
            default: return "en";
        }
    }
}
