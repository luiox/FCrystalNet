package com.github.luiox.fcrystalnet.poc.golden;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.github.luiox.fcrystalnet.poc.features.FeatureBuilder;
import com.github.luiox.fcrystalnet.poc.perception.MinimalPerception;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The features half of the golden chain: {@link FeatureBuilder} - the code the game runs
 * - must reproduce the python {@code features.py} tensors on real dataset frames to
 * 1e-3 (the bar covers the fp16 rounding of the distance fields and nothing else).
 * The frames arrive as raw .ncds payloads inside golden.bin, decoded by the Java
 * {@code read_frames} twin, so this test fails on any drift in channels, ordering,
 * normalisation, crop origin, or velocity recovery.
 */
class FeatureGoldenTest {

    private static final double TOLERANCE = 1e-3;
    private static final Path GOLDEN =
            Path.of("src", "test", "resources", "golden.bin");

    @Test
    void featuresMatchPython() throws Exception {
        List<GoldenFrames.Frame> frames = GoldenFrames.read(GOLDEN);
        assertTrue(frames.size() >= 4, "golden fixture too small: " + frames.size());
        for (int f = 0; f < frames.size(); f++) {
            GoldenFrames.Frame golden = frames.get(f);
            FrameDecoder.Decoded decoded = FrameDecoder.decode(golden.payload());
            MinimalPerception.Frame frame = FrameDecoder.toFrame(decoded);

            float[] x = FeatureBuilder.buildX(frame);
            assertClose("x", f, x, golden.x());

            float[] aux = FeatureBuilder.buildAux(frame);
            assertClose("aux", f, aux, golden.aux());
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
                () -> String.format("frame %d %s[%d] differs by %.6f (java %.6f, python %.6f)",
                        frame, tensor, worstAt, worstDiff,
                        actual[worstAt], expected[worstAt]));
    }
}
