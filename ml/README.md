# ml/ — 末影水晶放置 + 敲击的监督训练

给定周围地形和目标/自身实体状态，输出水晶放置格与敲击格。放置与敲击共用同一个伤害场头。纯 Python，不需要 Java/Gradle。依赖由 `pyproject.toml` + `uv.lock` 锁定（torch 2.11.0+cu128、numpy≥2），Python 3.12。

## 环境

安装 [uv](https://docs.astral.sh/uv/) 后：

```bash
cd ml
uv sync                 # 训练环境：torch + numpy（.venv，版本按 uv.lock 精确锁定）
```

设备自动选择：有 CUDA 用 CUDA（bf16 autocast 默认开），没有就用 CPU。强制 CPU 加 `--device cpu`，关 autocast 加 `--no-amp`。

导出和金样本生成还需要推理栈（onnx + onnxruntime），在可选依赖组里：

```bash
uv sync --group deploy
```

## 数据准备

数据集以 GitHub Releases 分包发布，不进 git。下载全部分包，解压到 `ml/data/ds`，目录结构：

```
ml/data/ds/
  shard_0000.ncds ...   # 分片帧文件
  meta.json             # 帧数等元信息
```

发布数据共 15000 帧。格式见 [docs/dataset-schema.md](../docs/dataset-schema.md)，读取按 magic + schemaVer 逐帧解码，文件截断后可恢复。

## 训练

```bash
# 正式配置（w24）：24 epochs
uv run python -u train.py --data data/ds --runs runs/w24v2gpu --epochs 24 --width 24

# 小模型对照（w12）：训练更快，精度见 eval 报告
uv run python -u train.py --data data/ds --runs runs/w12v2gpu --epochs 30 --width 12
```

要点：

- `--width` 是卷积宽度，w24 约 39.8k 参数，w12 约 12.6k；`--batch` 默认 256。
- 验证集按 seed 哈希切分（约 12%），与训练集无交叉。
- 伤害标签逐帧归一化（`norm_dmg`）：模型输出是分数不是半心，只用于排序；有效性和伤害由精确计算决定。见 [docs/reproducing.md](../docs/reproducing.md)。

### 断点续训

每个 epoch 结束写 `runs/<name>/ckpt.pt`，包含模型、优化器、调度器、RNG 状态和历史。中断后从同一条命令加 `--resume` 接着跑：

```bash
uv run python -u train.py --data data/ds --runs runs/w24v2gpu --epochs 24 --width 24 --resume
```

`--runs`、`--epochs`、`--width` 要和原命令一致。`model.pt` 是验证最优导出，任何机器可加载；续训用 `ckpt.pt`。

## 评估与导出

```bash
uv run python -u eval.py    --data data/ds --run runs/w24v2gpu   # top-k / regret / 分预设报告
uv run python -u deploy.py  --run  runs/w24v2gpu                 # 导出 model.onnx + ORT 延迟复测
uv run python -u quantize.py                                     # int8 量化实验（ORT 缺少 3D int8 内核，无加速）
```

- 评估主指标是 damage regret：teacher 最优伤害 − 模型所选格子的伤害。agreement@1 只作参考——对称局面下等价点位很多，它的天花板低。
- `deploy.py` 导出动态 batch 的 ONNX（opset 17），并给出 ORT 单线程/4 线程延迟。
- GPU 训练的权重在 CPU 机器上可直接加载（已做 map_location）。

## 金样本

```bash
uv run python -u dump_golden.py
```

从 `data/ds` 确定性抽 8 帧（6 帧含水晶 + 2 帧空场），按训练时的 fp16 舍入重建特征张量，跑 ORT 单线程推理，写出 `pocmod/src/test/resources/golden.bin`。之后 `gradlew :pocmod:test` 验证两条链（python 特征 == Java 特征、python ORT == DLL，均 ≤ 1e-3）。前置条件：`data/ds` 已就位、`runs/w24v2gpu/model.onnx` 已导出。

## 基准成绩

15000 帧（验证 1800），ORT 单线程 CPU 推理：

| 指标 | 数值 |
|---|---|
| 放置 top1 / top5（有效候选上） | 68.5% / 98.1% |
| regret 均值 / p90 | 0.77 / 0.70 半心 |
| 敲击 top1 / regret | 92.2% / 0.015 |

## 目录内容

```
features.py     帧 → 模型张量（x 11 通道 + aux 14 维；与 pocmod FeatureBuilder 逐行对应）
read_frames.py  .ncds 读取与解码（按帧头 magic + schemaVer）
model.py        CrystalNet：3D CNN（FiLM + bottleneck 注入 aux；dmg 头 Softplus，valid 头 logit）
train.py        训练（增广自检、断点续训、逐 epoch checkpoint）
eval.py         评估报告（top-k / regret 分位数 / 分预设 / 延迟）
deploy.py       ONNX 导出 + ORT 延迟复测
dump_golden.py  生成 pocmod 金样本 golden.bin
quantize.py     int8 量化实验
runs/           训练产物（gitignore）
data/           数据集（gitignore，从 Releases 下载，解压到 data/ds）
```
