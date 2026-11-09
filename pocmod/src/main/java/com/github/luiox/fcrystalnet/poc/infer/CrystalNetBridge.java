package com.github.luiox.fcrystalnet.poc.infer;

import com.github.luiox.fcrystalnet.poc.features.FeatureBuilder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * The native link: loads crystalnet.dll and runs one ONNX session inference per call.
 * The DLL embeds ONNX Runtime behind a dynamic load ({@code nativeInit} gets the
 * onnxruntime.dll path), so this side never links or loads ORT itself.
 *
 * <p>Wire conventions (WIRE_VERSION 1, mirrored in native/src/wire.h - the shapes are a
 * contract, not an implementation detail):
 *
 * <ul>
 *   <li>{@code x}: {@link FeatureBuilder#CHANNELS} &times; {@link FeatureBuilder#CELLS}
 *       f32 little endian = 52052 bytes, (channel, y, z, x) row-major</li>
 *   <li>{@code aux}: {@link FeatureBuilder#AUX_DIM} f32 = 56 bytes</li>
 *   <li>{@code dmg}, {@code valid}: {@link FeatureBuilder#CELLS} f32 each = 4732 bytes
 *       - the model's raw damage field and valid logit over the crop</li>
 * </ul>
 *
 * <p>All buffers are direct and reused write-once-per-cycle; nothing allocates on the
 * inference path. A version mismatch is a hard error - a silently mismatched boundary
 * would corrupt data, not fail fast.
 */
public final class CrystalNetBridge implements AutoCloseable {

    public static final int WIRE_VERSION = 1;

    public static final int CELLS = FeatureBuilder.CELLS;
    public static final int X_BYTES = FeatureBuilder.CHANNELS * CELLS * 4;
    public static final int AUX_BYTES = FeatureBuilder.AUX_DIM * 4;
    public static final int MAP_BYTES = CELLS * 4;

    /** {@code nativeInit} error codes, mirrored in native/src/bridge.cpp. */
    public static final int ERR_ORT_LOAD = 1;
    public static final int ERR_ORT_API = 2;
    public static final int ERR_MODEL_READ = 3;
    public static final int ERR_SESSION = 4;

    private static native int nativeVersion();

    /** Loads ORT dynamically and creates the session; returns an {@code ERR_*} code. */
    private static native int nativeInit(String ortPath, String modelPath);

    private static native void nativeInfer(ByteBuffer x, ByteBuffer aux,
            ByteBuffer dmg, ByteBuffer valid);

    private static native void nativeClose();

    private final ByteBuffer xBuf;
    private final ByteBuffer auxBuf;
    private final ByteBuffer dmgBuf;
    private final ByteBuffer validBuf;

    private CrystalNetBridge() {
        this.xBuf = direct(X_BYTES);
        this.auxBuf = direct(AUX_BYTES);
        this.dmgBuf = direct(MAP_BYTES);
        this.validBuf = direct(MAP_BYTES);
    }

    private static ByteBuffer direct(int size) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(size);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        return buffer;
    }

    /**
     * Loads the DLL, runs the wire handshake and initialises the embedded ORT session.
     * Throws with the failing stage named - the driver turns that into a chat line, the
     * tests turn it into a skip.
     */
    public static CrystalNetBridge acquire() {
        Path dll = materialize(locateDll(), "/native/crystalnet.dll", "crystalnet.dll");
        System.load(dll.toAbsolutePath().toString());
        int version = nativeVersion();
        if (version != WIRE_VERSION) {
            throw new IllegalStateException("crystalnet.dll wire version " + version
                    + " does not match java side " + WIRE_VERSION);
        }
        Path ort = materialize(locateOrt(), "/native/onnxruntime.dll", "onnxruntime.dll");
        Path model = materialize(locateModel(), "/model/model.onnx", "model.onnx");
        int err = nativeInit(ort.toAbsolutePath().toString(), model.toAbsolutePath().toString());
        if (err != 0) {
            throw new IllegalStateException("crystalnet nativeInit failed with code " + err
                    + " (ort=" + ort + ", model=" + model + ")");
        }
        return new CrystalNetBridge();
    }

    /**
     * One session run: {@code x} (52052 floats) and {@code aux} (14) in, the damage and
     * valid maps (1183 floats each) out.
     */
    public void infer(float[] x, float[] aux, float[] dmgOut, float[] validOut) {
        this.xBuf.clear();
        this.xBuf.asFloatBuffer().put(x);
        this.auxBuf.clear();
        this.auxBuf.asFloatBuffer().put(aux);
        nativeInfer(this.xBuf, this.auxBuf, this.dmgBuf, this.validBuf);
        this.dmgBuf.clear();
        this.dmgBuf.asFloatBuffer().get(dmgOut);
        this.validBuf.clear();
        this.validBuf.asFloatBuffer().get(validOut);
    }

    /** Releases the session and the ORT library; the bridge is dead afterwards. */
    @Override
    public void close() {
        nativeClose();
    }

    // --- path resolution -------------------------------------------------------------------

    /**
     * POC-only path resolution: runClient (FCrystalNet/run) and gradle tests (pocmod/)
     * both sit one level under the repo root, so "../native/..." covers both; the
     * "../../" shape stays as a belt for other launch layouts.
     */
    public static Path locateDll() {
        return locate("fcpoc.native.path", "../native/build/bin/crystalnet.dll",
                "../../native/build/bin/crystalnet.dll");
    }

    /** The ORT runtime the DLL loads dynamically. */
    public static Path locateOrt() {
        return locate("fcpoc.ort.path",
                "../native/third_party/onnxruntime/lib/onnxruntime.dll",
                "../../native/third_party/onnxruntime/lib/onnxruntime.dll");
    }

    /** The exported w24v2gpu ONNX model. */
    public static Path locateModel() {
        return locate("fcpoc.model.path", "../ml/runs/w24v2gpu/model.onnx",
                "../../ml/runs/w24v2gpu/model.onnx");
    }

    private static Path locate(String property, String... relatives) {
        String override = System.getProperty(property);
        if (override != null && !override.isBlank()) {
            return Paths.get(override).toAbsolutePath().normalize();
        }
        for (String relative : relatives) {
            Path candidate = Paths.get(relative).toAbsolutePath().normalize();
            if (Files.exists(candidate)) {
                return candidate;
            }
        }
        return Paths.get(relatives[0]).toAbsolutePath().normalize();
    }

    /**
     * Resolves one native artifact: the located dev-tree copy when it exists (runClient,
     * gradle tests), otherwise the copy bundled inside the jar, extracted into the
     * system temp dir - the distribution shape, where the stock launcher has no
     * FCrystalNet repo sitting next to the game. The copy runs on every acquire (one
     * toggle-on, ~20&nbsp;MB) so a refreshed jar can never be shadowed by a stale file.
     */
    private static Path materialize(Path located, String resource, String name) {
        if (Files.exists(located)) {
            return located;
        }
        Path target = cacheDir().resolve(name);
        try (java.io.InputStream in = CrystalNetBridge.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException(name + " not found at " + located
                        + " and not bundled in the jar (" + resource + ")");
            }
            Files.createDirectories(target.getParent());
            Files.copy(in, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (java.io.IOException e) {
            throw new IllegalStateException(
                    "extracting bundled " + name + " to " + target + " failed", e);
        }
        return target;
    }

    private static Path cacheDir() {
        return Paths.get(System.getProperty("java.io.tmpdir", "."), "fcpoc-native");
    }
}
