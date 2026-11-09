# 复现指南

从发布数据到游戏内 POC 的完整复现路径与验收标准。实现对应关系：训练与归一化在 `ml/train.py`，部署语义在 `pocmod` 的 `PocDecision`，live-regret harness 在 `PocDriver`，数据格式见 [dataset-schema.md](dataset-schema.md)。

## 基准 run：`runs/w24v2gpu`

3D CNN，三组卷积，中层 dilation=2，感受野 9，约 4 万参数。输入 `x (batch,11,7,13,13)` + `aux (batch,14)`，输出 `dmg` / `valid` 两张 `(batch,7,13,13)` 场。放置与敲击共用同一个 dmg 头：敲一颗格内水晶就是该格自己的爆炸，所以不需要单独的 break 头。

15000 帧合成数据，验证集 1800 帧按 seed 哈希切分。离线成绩，与 teacher（原版精确物理模型）对比，只在有效候选上统计：

| 指标 | 数值 |
|---|---|
| 放置 top1 / top5 | 68.5% / 98.1% |
| regret 均值 / p90 | 0.77 / 0.70 半心 |
| 敲击 top1 / regret | 92.2% / 0.015 |
| ORT 单线程延迟 | 6.9 ms |

## 部署语义

`train.py` 的 `norm_dmg` 在监督格（place ∪ breakable）上把每帧伤害场除以帧内最大伤害，所以 **dmg 头的输出是归一化分数，不是半心**——它编码的是伤害场的形状与排序。部署语义由这一点决定，实现见 `PocDecision`：

- **有效性完全由精确计算决定**：候选 = `CandidateStage.enumerate`（原版放置规则），再过滤眼距 4.5 和自伤检查（自伤 ≥ 血量 − 1 的格子直接排除）。
- **模型只在有效候选集内排序**；动作下限 2.0 半心施加在模型所选格子的**精确伤害**上。不要把模型分数当伤害用。
- 敲击收益 = dmg 场在水晶所在格的取值，不需要单独的敲击头。

驱动的机械节奏：先敲后放、两次爆炸间隔 ≥ 无敌窗、每 tick 至多一动作、不转头。见仓库根 README。

## 复现路径

### 一、数据

从 GitHub Releases 下载数据分包，解压到 `ml/data/ds`（下文命令都按这个路径），得到 `.ncds` 数据分卷与一份 `meta.json`（帧数、schemaVer、seed 范围）。

### 二、训练与离线评估

```bash
cd ml
uv sync                                  # torch + numpy（uv.lock 锁定版本）
uv run python -u train.py --data data/ds --runs runs/w24v2gpu --epochs 24 --width 24
uv run python -u eval.py  --data data/ds --run  runs/w24v2gpu
```

`eval.py` 输出上表的全部指标（含预设分层报告）。断点续训：同一条命令加 `--resume`（checkpoint 每 epoch 原子落盘）。

### 三、导出与原生部署

```bash
uv sync --group deploy                   # onnx / onnxruntime 属可选依赖组
uv run python -u deploy.py --run runs/w24v2gpu     # model.onnx + ORT 延迟复测
uv run python -u dump_golden.py                    # 重新生成 pocmod 金样本（可选）
```

原生 DLL 构建（xmake + MSVC + JDK 21）与 wire 协议见 [native/README.md](../native/README.md)。`gradlew :pocmod:build` 会把 `model.onnx` 与两个 DLL 一起打进 fat jar（存在才打包，见 pocmod/build.gradle）。

### 四、验收

**离线（每次训练必跑）**——`ml/eval.py`：

- 放置：有效候选上的 top1 / top5 / regret（均值 + p90）。主指标是 **damage regret 而非 agreement@1**：对称局面下等价点位很多，agreement 天花板低；regret 直接度量损失了多少半心。
- 敲击：breakable、距离、自伤过滤后的 top1 / regret。候选最多 0-3 格，没有 top5。

**在线（POC harness，每次换模型必跑）**——游戏内会话日志：

- 每个决策周期内，精确 argmax 与模型选择的格子对比，差值进日志；20 tick 输出一次 `ticks / actions / missed-opportunities / mean-regret` 摘要。
- 参考基线：静态假人 mean-regret 2.46，动态假人 4.15，0 次服务端拒绝。复现验收：同场景下同一量级。
- golden 回归随时可跑：`gradlew :pocmod:test`，python 特征 == Java 特征 == DLL 输出，容差 ≤1e-3，金样本随仓库走。

## 数据发布

数据集以 **GitHub Releases 分包**发布，不进 git。数据格式见 [dataset-schema.md](dataset-schema.md)，reader 按 magic + schemaVer 逐帧解码，文件截断或混版本都能安全处理。
