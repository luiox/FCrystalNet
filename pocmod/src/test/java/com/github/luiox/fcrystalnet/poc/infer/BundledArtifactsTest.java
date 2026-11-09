package com.github.luiox.fcrystalnet.poc.infer;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.Test;

/**
 * The distribution path of the native link: with the dev-tree overrides pointing at
 * nothing, {@link CrystalNetBridge#acquire()} must fall back to the copies bundled in
 * the jar (processResources puts them on this test's classpath too), extract them into
 * the temp dir, load the DLL, handshake and open the ORT session. This is exactly what
 * happens on a stock launcher's machine - exercised here so the fat jar can never
 * silently rot. Skipped on a bare clone, where nothing is built or exported yet.
 */
class BundledArtifactsTest {

    @Test
    void bundledArtifactsExtractAndLoadWhenDevTreeAbsent() {
        assumeTrue(CrystalNetBridge.class.getResource("/native/crystalnet.dll") != null
                        && CrystalNetBridge.class.getResource("/model/model.onnx") != null,
                "bundled artifacts absent - build native/ and export the model first");
        System.setProperty("fcpoc.native.path", "does-not-exist/crystalnet.dll");
        System.setProperty("fcpoc.ort.path", "does-not-exist/onnxruntime.dll");
        System.setProperty("fcpoc.model.path", "does-not-exist/model.onnx");
        try (CrystalNetBridge bridge = CrystalNetBridge.acquire()) {
            bridge.infer(new float[CrystalNetBridge.X_BYTES / 4],
                    new float[CrystalNetBridge.AUX_BYTES / 4],
                    new float[CrystalNetBridge.CELLS],
                    new float[CrystalNetBridge.CELLS]);
        } finally {
            System.clearProperty("fcpoc.native.path");
            System.clearProperty("fcpoc.ort.path");
            System.clearProperty("fcpoc.model.path");
        }
    }
}
