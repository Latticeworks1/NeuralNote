#include "WhisperModelLoader.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace WhisperModelLoader {

namespace fs = std::filesystem;

namespace {

std::vector<fs::path> gLastScannedDirs;

fs::path envPath(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return {};
    }
    return fs::path(value);
}

fs::path homePath()
{
#if defined(_WIN32)
    auto home = envPath("USERPROFILE");
    if (!home.empty()) {
        return home;
    }
    return envPath("HOMEDRIVE") / envPath("HOMEPATH");
#else
    return envPath("HOME");
#endif
}

bool readFile(const fs::path& filePath, std::vector<uint8_t>& buffer, std::string& error)
{
    std::ifstream stream(filePath, std::ios::binary | std::ios::ate);
    if (!stream.is_open()) {
        error = "Failed to open file: " + filePath.string();
        return false;
    }

    const std::streamsize size = stream.tellg();
    if (size <= 0) {
        error = "File is empty: " + filePath.string();
        return false;
    }

    stream.seekg(0, std::ios::beg);
    buffer.resize(static_cast<size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(buffer.data()), size)) {
        error = "Failed to read file: " + filePath.string();
        buffer.clear();
        return false;
    }

    return true;
}

std::vector<fs::path> candidateDirectories()
{
    std::vector<fs::path> dirs;
    if (auto env = envPath("NEURALNOTE_WHISPER_DIR"); !env.empty()) {
        dirs.push_back(env);
    }

    const auto home = homePath();
    if (!home.empty()) {
        dirs.push_back(home / ".neuralnote" / "models");
#if defined(__APPLE__)
        dirs.push_back(home / "Library" / "Application Support" / "NeuralNote" / "Models");
#elif defined(_WIN32)
        if (auto appData = envPath("APPDATA"); !appData.empty()) {
            dirs.push_back(appData / "NeuralNote" / "Models");
        }
#else
        if (auto xdg = envPath("XDG_DATA_HOME"); !xdg.empty()) {
            dirs.push_back(xdg / "NeuralNote" / "Models");
        } else {
            dirs.push_back(home / ".local" / "share" / "NeuralNote" / "Models");
        }
#endif
    }

    dirs.push_back(fs::current_path() / "NeuralNoteModels");
    dirs.push_back(fs::current_path() / "Lib" / "ModelData");

    // Deduplicate paths while preserving order
    std::vector<fs::path> unique;
    for (const auto& dir: dirs) {
        if (dir.empty()) {
            continue;
        }
        const auto canonicalPath = dir.lexically_normal();
        const bool already = std::any_of(unique.begin(), unique.end(), [&](const fs::path& existing) {
            return existing == canonicalPath;
        });
        if (!already) {
            unique.push_back(canonicalPath);
        }
    }

    return unique;
}

LoadResult loadInternal(const fs::path& directory, std::vector<uint8_t>& outEncoder, std::vector<uint8_t>& outDecoder)
{
    LoadResult result;
    if (directory.empty()) {
        result.message = "Invalid directory supplied for Whisper models.";
        return result;
    }

    const fs::path encoderPath = directory / "whisper_encoder.ort";
    const fs::path decoderPath = directory / "whisper_decoder.ort";

    if (!fs::exists(encoderPath) || !fs::exists(decoderPath)) {
        std::ostringstream oss;
        oss << "Missing model files inside " << directory;
        result.message = oss.str();
        return result;
    }

    std::string error;
    if (!readFile(encoderPath, outEncoder, error)) {
        result.message = error;
        outEncoder.clear();
        outDecoder.clear();
        return result;
    }

    if (!readFile(decoderPath, outDecoder, error)) {
        result.message = error;
        outEncoder.clear();
        outDecoder.clear();
        return result;
    }

    result.success = true;
    std::ostringstream oss;
    oss << "Loaded Whisper models from " << directory;
    result.message = oss.str();
    return result;
}

} // namespace

LoadResult loadFromDirectory(const fs::path& directory,
                             std::vector<uint8_t>& outEncoder,
                             std::vector<uint8_t>& outDecoder)
{
    return loadInternal(directory, outEncoder, outDecoder);
}

LoadResult loadFromDefaultLocations(std::vector<uint8_t>& outEncoder, std::vector<uint8_t>& outDecoder)
{
    gLastScannedDirs = candidateDirectories();

    for (const auto& directory: gLastScannedDirs) {
        auto result = loadInternal(directory, outEncoder, outDecoder);
        if (result.success) {
            return result;
        }
    }

    std::ostringstream oss;
    oss << "Whisper models not embedded and no external files were found.\n"
        << "Place whisper_encoder.ort and whisper_decoder.ort into one of the following directories or set "
           "NEURALNOTE_WHISPER_DIR:\n";
    for (const auto& dir: gLastScannedDirs) {
        oss << "  - " << dir << '\n';
    }
    LoadResult result;
    result.success = false;
    result.message = oss.str();
    return result;
}

const std::vector<fs::path>& getLastScannedDirectories()
{
    return gLastScannedDirs;
}

} // namespace WhisperModelLoader
