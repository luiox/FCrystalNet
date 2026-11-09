package com.github.luiox.fcrystalnet.poc.golden;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.github.luiox.fcrystalnet.poc.features.FeatureBuilder;
import com.github.luiox.fcrystalnet.poc.infer.CrystalNetBridge;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The inference half of the golden chain: crystalnet.dll's embedded ONNX session must
 * reproduce the python ORT outputs on the golden tensors to 1e-3. Skips (does not fail)
 * when the DLL, onnxruntime.dll or the model is absent, so a clean checkout without the native artifacts still
 * tests green.
 */
class InferGoldenTest {

    private static final double TOLERANCE = 1e-3;
    private static final Path GOLDEN =
            Path.of("src", "test", "resources", "golden.bin");

    @Test
    void inferenceMatchesOrt() throws Exception {
        List<GoldenFrames.Frame> frames = GoldenFrames.read(GOLDEN);
        CrystalNetBridge acquired = null;
        Throwable failure = null;
        try {
            acquired = CrystalNetBridge.acquire();
        } catch (Throwable t) {
            failure = t;
        }
        assumeTrue(acquired != null, "native inference unavailable: " + failure);
        try (CrystalNetBridge bridge = acquired) {
            float[] dmgOut = new float[FeatureBuilder.CELLS];
            float[] validOut = new float[FeatureBuilder.CELLS];
            for (int f = 0; f < frames.size(); f++) {
                GoldenFrames.Frame golden = frames.get(f);
                bridge.infer(golden.x(), golden.aux(), dmgOut, validOut);
                assertClose("dmg", f, dmgOut, golden.dmg());
                assertClose("valid", f, validOut, golden.valid());
            }
        }
    }

    private static void assertClose(String tensor, int frame, float[] actual,
                                    float[] expected) {
        double worst = 0.0;
        int worstIndex = -1;
        for (int i = 0; i < expected.length; i++) {
            double diff = Math.abs(actual[i] - expected[i]);
            if (diff > worst) {
                worst = diff;
                worstIndex = i;
            }
        }
        double worstDiff = worst;
        int worstAt = worstIndex;
        assertTrue(worst <= TOLERANCE,
                () -> String.format("frame %d %s[%d] differs by %.6f (dll %.6f, ort %.6f)",
                        frame, tensor, worstAt, worstDiff,
                        actual[worstAt], expected[worstAt]));
    }
}
