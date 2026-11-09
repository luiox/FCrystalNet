# FCrystalNet

English | [简体中文](README.zh-CN.md)

A neural-network research framework for end-crystal PvP in Minecraft 1.21.11 (Fabric): dataset format, training code, a native inference DLL, and an in-game prototype mod.

## How it fits together

- **Offline** (`ml/`, Python): read frames → build features → train → export ONNX
- **In-game** (`pocmod/` + `native/`): perceive → build the same features → crystalnet.dll inference → decision → real crystal placements and attacks, with a live-regret log comparing the model against exact search every cycle

## What's here

- **Dataset format**: the `.ncds` frame format, magic + schemaVer per frame, readable even after a truncated file. Documented in [docs/dataset-schema.md](docs/dataset-schema.md). Datasets ship as GitHub release assets.
- **Training**: synthetic scenes densely labelled by a teacher — the exact vanilla physics model. A small 3D CNN (about 40k parameters) learns one damage field shared by placement and attack.
- **Native inference**: crystalnet.dll embeds ONNX Runtime 1.31 behind a versioned JNI boundary. 1-7 ms single-threaded. ORT is loaded at runtime; the DLL itself has no link-time dependencies.
- **In-game prototype**: a mixin-free fabric mod. G toggles it. Real placements and attacks, no camera movement — vanilla checks range, not view direction.
- **Live regret**: every decision cycle also computes the exact best answer; the gap between the model's pick and that answer goes to the log. Deployment quality is measurable.
- **Golden tests** pin the whole chain: python features == Java features == DLL outputs, within 1e-3. One command: `gradlew :pocmod:test`.

## Benchmark

Run `runs/w24v2gpu`, trained on 15000 synthetic frames (1800 validation, seed-hashed split):

| metric | value |
|---|---|
| placement top5 | 98.1% |
| mean regret vs exact search | 0.77 half-hearts |
| attack top1 | 92.2% |

In-game: mean-regret 2.46 with a static fake player, 4.15 with a patrolling one; zero server rejections.

The damage head is trained per-frame normalized, so model outputs are **scores, not damage values**. Validity is decided entirely by exact computation — vanilla placement rules, 4.5 reach, a self-damage veto. The model only ranks valid candidates, and the action floor applies to the picked cell's exact damage. See [docs/reproducing.md](docs/reproducing.md).

## Quick start

**Training** — download the dataset from Releases, unzip to `ml/data/ds`, then:

```bash
cd ml
uv sync
uv run python -u train.py --data data/ds --runs runs/w24v2gpu --epochs 24 --width 24
uv run python -u eval.py  --data data/ds --run  runs/w24v2gpu
```

**Native DLL** — needs xmake, MSVC and a JDK 21; details in [native/README.md](native/README.md):

```bash
cd native && xmake f -m release && xmake
```

**In-game prototype**:

```bash
gradlew :pocmod:runClient     # dev mode
gradlew :pocmod:build         # fat jar: pocmod/build/libs/fcpoc-0.1.0.jar
```

The fat jar, about 9.5 MB, bundles fabric-api, both DLLs and the model. On a stock launcher you only need the fabric loader — drop the jar into `mods`. In a world: build an obsidian platform extending at least two blocks beyond the target, `.fcn fakeplayer place` spawns a dummy, G toggles the prototype. Full reproduction steps and acceptance criteria in [docs/reproducing.md](docs/reproducing.md).

## Layout

```
FCrystalNet/
├─ pocmod/     the fabric mod: perception / features / JNI inference / decisions / actions / dummies
├─ ml/         training: frame reader, features, model, train/eval/export/golden (Python, uv)
├─ native/     crystalnet.dll: an ORT session + four JNI exports (xmake / MSVC)
└─ docs/       reproduction guide, dataset format
```

## License

MIT, see [LICENSE](LICENSE).
