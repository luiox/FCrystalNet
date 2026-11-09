#include "infer.h"

#include "wire.h"

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#undef ORT_API_MANUAL_INIT

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace fc {
namespace infer {
namespace {

const OrtApi* g_ort = nullptr;
OrtEnv* g_env = nullptr;
OrtSessionOptions* g_options = nullptr;
OrtSession* g_session = nullptr;

// The shapes are fixed by the wire, so every tensor is allocated once over a persistent
// buffer and reused - the inference path never allocates.
std::vector<float> g_xBuf(wire::kChannels * wire::kCells);
std::vector<float> g_auxBuf(wire::kAuxDim);
std::vector<float> g_dmgBuf(wire::kCells);
std::vector<float> g_validBuf(wire::kCells);
OrtValue* g_inputX = nullptr;
OrtValue* g_inputAux = nullptr;
OrtValue* g_outputDmg = nullptr;
OrtValue* g_outputValid = nullptr;
const char* g_inNames[2] = {nullptr, nullptr};
const char* g_outNames[2] = {nullptr, nullptr};

// One-time init/run guards: the driver calls from the client thread only, but the JVM
// may run tests concurrently with the game - the lock is cheap and removes the question.
std::mutex g_mutex;

std::wstring toWide(const std::string& utf8) {
    if (utf8.empty()) {
        return std::wstring();
    }
    int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(),
                                static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()),
                        out.data(), n);
    return out;
}

OrtValue* makeTensor(OrtMemoryInfo* mem, float* data, size_t floats,
                     const int64_t* shape, size_t rank) {
    OrtValue* value = nullptr;
    OrtStatus* st = g_ort->CreateTensorWithDataAsOrtValue(
            mem, data, floats * sizeof(float), shape, rank,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value);
    if (st != nullptr) {
        g_ort->ReleaseStatus(st);
        return nullptr;
    }
    return value;
}

}  // namespace

int init(const std::string& ortPath, const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_session != nullptr) {
        return 0;  // idempotent: the JVM may re-acquire after a close
    }
    HMODULE library = LoadLibraryW(toWide(ortPath).c_str());
    if (library == nullptr) {
        return kErrOrtLoad;
    }
    using GetApiBaseFn = const OrtApiBase* (*)();
    auto getBase = reinterpret_cast<GetApiBaseFn>(
            GetProcAddress(library, "OrtGetApiBase"));
    if (getBase == nullptr) {
        return kErrOrtLoad;
    }
    g_ort = getBase()->GetApi(ORT_API_VERSION);
    if (g_ort == nullptr) {
        return kErrOrtApi;
    }

    if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "crystalnet", &g_env) != nullptr) {
        return kErrSession;
    }
    // Fail before the session machinery if the model file itself is unreadable.
    FILE* probe = nullptr;
    if (_wfopen_s(&probe, toWide(modelPath).c_str(), L"rb") != 0 || probe == nullptr) {
        return kErrModelRead;
    }
    fclose(probe);

    if (g_ort->CreateSessionOptions(&g_options) != nullptr) {
        return kErrSession;
    }
    // The inference budget lives on one core beside Minecraft's render thread.
    g_ort->SetIntraOpNumThreads(g_options, 1);
    g_ort->SetInterOpNumThreads(g_options, 1);
    if (g_ort->CreateSession(g_env, toWide(modelPath).c_str(), g_options,
                             &g_session) != nullptr) {
        return kErrSession;
    }

    // IO names come from the session, not from assumptions about the exporter.
    OrtAllocator* allocator = nullptr;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    char* name = nullptr;
    if (g_ort->SessionGetInputName(g_session, 0, allocator, &name) == nullptr) {
        g_inNames[0] = name;
    }
    if (g_ort->SessionGetInputName(g_session, 1, allocator, &name) == nullptr) {
        g_inNames[1] = name;
    }
    if (g_ort->SessionGetOutputName(g_session, 0, allocator, &name) == nullptr) {
        g_outNames[0] = name;
    }
    if (g_ort->SessionGetOutputName(g_session, 1, allocator, &name) == nullptr) {
        g_outNames[1] = name;
    }
    if (g_inNames[0] == nullptr || g_inNames[1] == nullptr
            || g_outNames[0] == nullptr || g_outNames[1] == nullptr) {
        return kErrSession;
    }
    size_t inCount = 0;
    size_t outCount = 0;
    g_ort->SessionGetInputCount(g_session, &inCount);
    g_ort->SessionGetOutputCount(g_session, &outCount);
    fprintf(stderr, "[crystalnet] session io: in=%zu (%s, %s) out=%zu (%s, %s)\n",
            inCount, g_inNames[0], g_inNames[1], outCount, g_outNames[0], g_outNames[1]);
    fflush(stderr);

    OrtMemoryInfo* mem = nullptr;
    if (g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem)
            != nullptr) {
        return kErrSession;
    }
    constexpr int64_t kXShape[5] = {1, wire::kChannels, wire::kDimY, wire::kDimZ,
                                    wire::kDimX};
    constexpr int64_t kAuxShape[2] = {1, wire::kAuxDim};
    // The exporter keeps the spatial dims on the outputs: dmg/valid come out as
    // (batch, 7, 13, 13) - rank 4 - and the flat row-major layout is identical.
    constexpr int64_t kOutShape[4] = {1, wire::kDimY, wire::kDimZ, wire::kDimX};
    g_inputX = makeTensor(mem, g_xBuf.data(), g_xBuf.size(), kXShape, 5);
    g_inputAux = makeTensor(mem, g_auxBuf.data(), g_auxBuf.size(), kAuxShape, 2);
    g_outputDmg = makeTensor(mem, g_dmgBuf.data(), g_dmgBuf.size(), kOutShape, 4);
    g_outputValid = makeTensor(mem, g_validBuf.data(), g_validBuf.size(), kOutShape, 4);
    g_ort->ReleaseMemoryInfo(mem);
    if (g_inputX == nullptr || g_inputAux == nullptr || g_outputDmg == nullptr
            || g_outputValid == nullptr) {
        return kErrSession;
    }
    return 0;
}

void run(const float* x, const float* aux, float* dmg, float* valid) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_session == nullptr) {
        std::memset(dmg, 0, wire::kMapBytes);
        std::memset(valid, 0, wire::kMapBytes);
        return;
    }
    std::memcpy(g_xBuf.data(), x, wire::kXBytes);
    std::memcpy(g_auxBuf.data(), aux, wire::kAuxBytes);

    const OrtValue* inputs[2] = {g_inputX, g_inputAux};
    OrtValue* outputs[2] = {g_outputDmg, g_outputValid};
    // 1.31's Run takes (outNames, outLen, outValues) - the length moved before the
    // array compared to the pre-1.31 order.
    OrtStatus* st = g_ort->Run(g_session, nullptr, g_inNames, inputs, 2, g_outNames, 2,
                               outputs);
    if (st != nullptr) {
        const char* msg = g_ort->GetErrorMessage(st);
        fprintf(stderr, "[crystalnet] run failed: %s\n", msg != nullptr ? msg : "?");
        fflush(stderr);
        g_ort->ReleaseStatus(st);
        std::memset(dmg, 0, wire::kMapBytes);
        std::memset(valid, 0, wire::kMapBytes);
        return;
    }
    std::memcpy(dmg, g_dmgBuf.data(), wire::kMapBytes);
    std::memcpy(valid, g_validBuf.data(), wire::kMapBytes);
}

void shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_inputX != nullptr) {
        g_ort->ReleaseValue(g_inputX);
        g_inputX = nullptr;
    }
    if (g_inputAux != nullptr) {
        g_ort->ReleaseValue(g_inputAux);
        g_inputAux = nullptr;
    }
    if (g_outputDmg != nullptr) {
        g_ort->ReleaseValue(g_outputDmg);
        g_outputDmg = nullptr;
    }
    if (g_outputValid != nullptr) {
        g_ort->ReleaseValue(g_outputValid);
        g_outputValid = nullptr;
    }
    if (g_session != nullptr) {
        g_ort->ReleaseSession(g_session);
        g_session = nullptr;
    }
    if (g_options != nullptr) {
        g_ort->ReleaseSessionOptions(g_options);
        g_options = nullptr;
    }
    if (g_env != nullptr) {
        g_ort->ReleaseEnv(g_env);
        g_env = nullptr;
    }
}

}  // namespace infer
}  // namespace fc
