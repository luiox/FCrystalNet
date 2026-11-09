// One embedded ONNX Runtime session, fully dynamically loaded: crystalnet.dll carries no
// ORT import - the onnxruntime.dll path arrives from the JVM at init time.
#pragma once

#include <string>

namespace fc {
namespace infer {

// Error codes, mirrored in CrystalNetBridge.java.
constexpr int kErrOrtLoad = 1;
constexpr int kErrOrtApi = 2;
constexpr int kErrModelRead = 3;
constexpr int kErrSession = 4;

// Loads onnxruntime.dll (UTF-8 path), creates the fixed-shape session, pre-allocates the
// IO tensors. All paths are UTF-8; converted to wide on this side.
int init(const std::string& ortPath, const std::string& modelPath);

// One session run: x (kChannels*kCells f32, LE) + aux (kAuxDim f32) into dmg/valid
// (kCells f32 each). A failed run zero-fills the outputs - a cycle that errors must not
// reuse stale damage numbers as if they were fresh.
void run(const float* x, const float* aux, float* dmg, float* valid);

// Releases the session and the tensors; the library handle stays with the process.
void shutdown();

}  // namespace infer
}  // namespace fc
