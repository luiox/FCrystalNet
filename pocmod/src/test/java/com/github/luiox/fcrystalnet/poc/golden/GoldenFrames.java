package com.github.luiox.fcrystalnet.poc.golden;

import com.github.luiox.fcrystalnet.poc.features.FeatureBuilder;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * The {@code golden.bin} reader: the fixture {@code ml/dump_golden.py} wrote, one record
 * per picked frame - the raw .ncds payload plus the python side's feature tensors and ORT
 * outputs. Everything little endian.
 */
final class GoldenFrames {

    /** One golden record: payload for {@link FrameDecoder}, tensors for comparison. */
    record Frame(byte[] payload, float[] x, float[] aux, float[] dmg, float[] valid) {
    }

    private static final byte[] MAGIC = {'F', 'C', 'G', '1'};
    private static final int X_FLOATS = FeatureBuilder.CHANNELS * FeatureBuilder.CELLS;
    private static final int AUX_FLOATS = FeatureBuilder.AUX_DIM;
    private static final int MAP_FLOATS = FeatureBuilder.CELLS;

    private GoldenFrames() {
    }

    static List<Frame> read(Path path) throws IOException {
        ByteBuffer p = ByteBuffer.wrap(Files.readAllBytes(path))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (byte b : MAGIC) {
            if (p.get() != b) {
                throw new IOException("bad golden magic in " + path);
            }
        }
        int count = p.getInt();
        List<Frame> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int len = p.getInt();
            byte[] payload = new byte[len];
            p.get(payload);
            out.add(new Frame(payload, floats(p, X_FLOATS), floats(p, AUX_FLOATS),
                    floats(p, MAP_FLOATS), floats(p, MAP_FLOATS)));
        }
        if (p.hasRemaining()) {
            throw new IOException("golden trailing bytes: " + p.remaining());
        }
        return out;
    }

    private static float[] floats(ByteBuffer p, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = p.getFloat();
        }
        return out;
    }
}
