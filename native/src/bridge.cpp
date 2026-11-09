// The JNI boundary: four exports, POD through direct ByteBuffers, paths as jstring.
// Symbol names mirror the native methods of
// pocmod/src/main/java/com/github/luiox/fcrystalnet/poc/infer/CrystalNetBridge.java -
// a package or class rename must land on both sides in the same commit.
#include <jni.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "infer.h"
#include "wire.h"

extern "C" JNIEXPORT jint JNICALL
Java_com_github_luiox_fcrystalnet_poc_infer_CrystalNetBridge_nativeVersion(JNIEnv*, jclass) {
    return fc::wire::kVersion;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_github_luiox_fcrystalnet_poc_infer_CrystalNetBridge_nativeInit(
        JNIEnv* env, jclass, jstring ortPath, jstring modelPath) {
    if (ortPath == nullptr || modelPath == nullptr) {
        return fc::infer::kErrModelRead;
    }
    const char* ort = env->GetStringUTFChars(ortPath, nullptr);
    const char* model = env->GetStringUTFChars(modelPath, nullptr);
    std::string ortStr = ort != nullptr ? ort : "";
    std::string modelStr = model != nullptr ? model : "";
    if (ort != nullptr) {
        env->ReleaseStringUTFChars(ortPath, ort);
    }
    if (model != nullptr) {
        env->ReleaseStringUTFChars(modelPath, model);
    }
    return fc::infer::init(ortStr, modelStr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_github_luiox_fcrystalnet_poc_infer_CrystalNetBridge_nativeInfer(
        JNIEnv* env, jclass, jobject x, jobject aux, jobject dmg, jobject valid) {
    auto* px = static_cast<const float*>(env->GetDirectBufferAddress(x));
    auto* paux = static_cast<const float*>(env->GetDirectBufferAddress(aux));
    auto* pdmg = static_cast<float*>(env->GetDirectBufferAddress(dmg));
    auto* pvalid = static_cast<float*>(env->GetDirectBufferAddress(valid));
    if (px == nullptr || paux == nullptr || pdmg == nullptr || pvalid == nullptr) {
        return;
    }
    // The buffers are direct LE f32 by construction; the byte contract still holds.
    if (env->GetDirectBufferCapacity(x) < static_cast<jlong>(fc::wire::kXBytes)
            || env->GetDirectBufferCapacity(aux) < static_cast<jlong>(fc::wire::kAuxBytes)
            || env->GetDirectBufferCapacity(dmg) < static_cast<jlong>(fc::wire::kMapBytes)
            || env->GetDirectBufferCapacity(valid) < static_cast<jlong>(fc::wire::kMapBytes)) {
        return;
    }
    fc::infer::run(px, paux, pdmg, pvalid);
}

extern "C" JNIEXPORT void JNICALL
Java_com_github_luiox_fcrystalnet_poc_infer_CrystalNetBridge_nativeClose(JNIEnv*, jclass) {
    fc::infer::shutdown();
}
