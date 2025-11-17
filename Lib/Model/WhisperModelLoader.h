#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

/**
 * Helper utilities to locate Whisper ONNX model files on disk when they are not
 * embedded via BinaryData.
 */
namespace WhisperModelLoader {

struct LoadResult {
    bool success = false;
    std::string message;
};

/**
 * Attempt to load whisper_encoder.ort and whisper_decoder.ort from a list of
 * default directories (environment variables, application data folders, and
 * repository-relative paths).
 *
 * @param outEncoder Buffer to fill with encoder bytes on success.
 * @param outDecoder Buffer to fill with decoder bytes on success.
 * @return Result structure detailing success and any error information.
 */
LoadResult loadFromDefaultLocations(std::vector<uint8_t>& outEncoder, std::vector<uint8_t>& outDecoder);

/**
 * Attempt to load models from a specific directory.
 *
 * @param directory Directory that should contain the .ort files.
 */
LoadResult loadFromDirectory(const std::filesystem::path& directory,
                             std::vector<uint8_t>& outEncoder,
                             std::vector<uint8_t>& outDecoder);

/**
 * Absolute paths that were inspected the last time loadFromDefaultLocations()
 * was invoked. Useful for diagnostics in the UI.
 */
const std::vector<std::filesystem::path>& getLastScannedDirectories();

} // namespace WhisperModelLoader
