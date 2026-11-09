# FCrystalNet

[English](README.md) | 简体中文

Minecraft 1.21.11（Fabric）末影水晶 PvP 的神经网络研究框架：数据格式、训练代码、原生推理 DLL 和游戏内原型 mod。

## 整体结构

- **离线**（`ml/`，Python）：读帧 → 构建特征 → 训练 → 导出 ONNX
- **游戏内**（`pocmod/` + `native/`）：感知 → 构建同样的特征 → crystalnet.dll 推理 → 决策 → 真实放置/敲击，每个决策周期同时用精确搜索对模型打分，差值进日志

## 内容

- **数据格式**：`.ncds` 帧格式，每帧带 magic 和 schemaVer，文件截断后仍可读取。见 [docs/dataset-schema.md](docs/dataset-schema.md)。数据集以 GitHub Releases 分包发布。
- **训练**：合成场景由 teacher 稠密打标——teacher 就是原版精确物理模型。一个约 4 万参数的小 3D CNN 学习一张放置和敲击共用的伤害场。
- **原生推理**：crystalnet.dll 内嵌 ONNX Runtime 1.31，JNI 边界带版本握手。单线程 1-7 ms。ORT 运行时加载，DLL 本身没有链接期依赖。
- **游戏内原型**：零 mixin 的 fabric mod，G 键开关。真实放置和敲击，不转头——原版只校验距离，不校验视角。
- **live regret**：每个决策周期同时计算精确最优解，模型选择与它的差值进日志。部署质量是可测量的数字。
- **golden 测试**覆盖全链：python 特征 == Java 特征 == DLL 输出，容差 1e-3。一条命令：`gradlew :pocmod:test`。

## 基准结果

run `runs/w24v2gpu`，15000 帧合成数据训练（验证集 1800 帧，按 seed 哈希切分）：

| 指标 | 数值 |
|---|---|
| 放置 top5 | 98.1% |
| 对精确搜索的平均 regret | 0.77 半心 |
| 敲击 top1 | 92.2% |

游戏内实测：静态假人 mean-regret 2.46，巡逻假人 4.15，0 次服务端拒绝。

伤害头按帧归一化训练，模型输出是**分数，不是伤害值**。有效性完全由精确计算决定——原版放置规则、4.5 格距离、自伤检查。模型只在有效候选里排序，动作下限施加在所选格子的精确伤害上。见 [docs/reproducing.md](docs/reproducing.md)。

## 快速开始

**训练**——从 Releases 下载数据集，解压到 `ml/data/ds`，然后：

```bash
cd ml
uv sync
uv run python -u train.py --data data/ds --runs runs/w24v2gpu --epochs 24 --width 24
uv run python -u eval.py  --data data/ds --run  runs/w24v2gpu
```

**原生 DLL**——需要 xmake、MSVC 和 JDK 21，细节见 [native/README.md](native/README.md)：

```bash
cd native && xmake f -m release && xmake
```

**游戏内原型**：

```bash
gradlew :pocmod:runClient     # 开发形态
gradlew :pocmod:build         # fat jar：pocmod/build/libs/fcpoc-0.1.0.jar
```

fat jar 约 9.5 MB，内嵌 fabric-api、两个 DLL 和模型。原版启动器只需要装 fabric loader，把 jar 丢进 `mods` 即可。进世界后：铺黑曜石台（至少超出目标两格），`.fcn fakeplayer place` 生成假人，G 键开关原型。完整复现步骤和验收标准见 [docs/reproducing.md](docs/reproducing.md)。

## 目录

```
FCrystalNet/
├─ pocmod/     fabric mod：感知 / 特征 / JNI 推理 / 决策 / 动作 / 假人
├─ ml/         训练：读帧、特征、模型、训练/评估/导出/金样本（Python, uv）
├─ native/     crystalnet.dll：一个 ORT 会话 + 四个 JNI 导出（xmake / MSVC）
└─ docs/       复现指南、数据格式
```

## 协议

MIT，见 [LICENSE](LICENSE)。
