# `.ncds` 数据格式

本文说明 `.ncds` 数据在磁盘上的帧布局，以及训练特征张量的定义。读取实现在 `ml/read_frames.py`；特征实现在 `ml/features.py` 与 `pocmod` 的 `FeatureBuilder.java`，两边由 golden 测试保证一致（见第四节）。数据集以 GitHub Releases 分包发布。

格式只存原始数据：方块序号、实体属性、预测点列表。特征在读取侧计算，改特征不需要重新发布数据。每帧带 magic、schemaVer 和长度，文件截断后能从下一条记录恢复读取。当前数据的 schemaVer = 2；更早的内部版本从未发布过。

## 一、磁盘帧布局

一个 `.ncds` 文件是若干帧的顺序拼接，配套一个 `meta.json`。每帧：

```
magic "NCDS" (4B) | u32 schemaVer (=2, 小端) | u32 payloadLen (小端) | payload (payloadLen B)
```

全部小端。payload 布局：

| 字段 | 类型 | 说明 |
|---|---|---|
| `seed` | i64 | 场景种子，也是验证集切分的哈希输入 |
| `preset` | u8 | 场景预设：0=open_field, 1=box_fight, 2=hole, 3=nether_roof, 4=pillar_city, 5=blobs, 6=mixed |
| `difficulty` | u8 | 游戏难度 0-3，aux 里除以 3 |
| `aim_steps` | u8 | teacher（原版精确物理模型）打标用的预测步数，aux 里除以 5 |
| `ent_count` | u8 | 实体条数 |
| `tvel` | f32×3 | 目标速度，每 tick 格数 |
| `svel` | f32×3 | 自身速度 |
| `t_armor, t_tough, t_prot, t_health` | f32×4 | 目标护甲、韧性、保护等级、血量 |
| `s_armor, s_tough, s_prot, s_health` | f32×4 | 自身同四项 |
| `origin` | i32×3 | crop 原点世界坐标 = 目标脚格 + (-6, -3, -6) |
| `dims` | u8×3 | (sx, sy, sz) = (13, 7, 13)，即 X±6、Y[-3,+3]，相对目标脚格 |

实体条目，`ent_count` 个顺序拼接：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | u8 | 0=self, 1=target, 2=other, **3=end crystal** |
| `pos` | f64×3 | 实体位置 |
| `box` | f64×6 | 碰撞箱 min/max XYZ；水晶按原版 Box(-1,0,-1,1,2,1) 偏移 |
| `stats` | f32×6 | 顺序固定：health, maxHealth, armor, toughness, protection, resistanceLevel |
| `npred` | u8 | 预测点数 |
| `preds` | f64×3 × npred | 预测位置列表，preds[i] = pos + vel × i × stepTicks；最后一个预测点决定 aim 通道和 aux 速度 |

标签网格与 crop 同网格，`cells = 13×7×13 = 1183`，全部按 (y, z, x) 行主序展平：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | u8 × cells | 地形种类，即 BlockSnapshot.Kind 序号：AIR=0, REPLACEABLE=1, OBSIDIAN=2, BEDROCK=3, OTHER=4。此序号就是磁盘字节，不可重排 |
| `valid` | u8 × cells | 现有底座可放的格子，teacher 用 CandidateStage.enumerate 且 allowBase=false 得出 |
| `needs_base` | u8 × cells | schemaVer=2 起恒为 0，布局保留给未来版本 |
| `reach` | u8 × cells | 距 self 眼睛 ≤ 4.5 的格子 |
| `breakable` | u8 × cells | schemaVer=2 起新增：该格有水晶实体且在目标 12 格交战圈内；crop 外的水晶直接丢弃 |
| `dmg` | f32 × cells | 全格伤害场，由 DamageModel 算出。爆心 = 格底面中心 (x+0.5, y, z+0.5)，目标按预测位置打分。放置与敲击共用 |
| `self_dmg` | f32 × cells | 自伤场：当前位置与全部预测位置取最大 |

reader 末尾断言 payload 恰好用完，多一字节少一字节都报错。

## 二、特征张量定义

训练侧 `ml/features.py` 与游戏内 `FeatureBuilder.java` 逐行对应，golden 测试保证两边一致（见第四节）。crop 以目标为中心，目标位置从 aim 通道直接读出。

### x：(11, 7, 13, 13)，通道主序 (channel, y, z, x)，存储 fp16

| 通道 | 名字 | 定义 |
|---|---|---|
| 0-4 | 地形 one-hot | kind 网格的 5 类 one-hot |
| 5 | `CH_AIM` | 目标碰撞箱在预测位置上的距离场：aim 箱 = 当前箱平移"最后一个预测点 − 当前位置" |
| 6 | `CH_CUR` | 目标碰撞箱当前位置的距离场 |
| 7 | `CH_SELF` | 自身箱的距离场，按 pos 造 0.6×1.8 箱 |
| 8 | `CH_OTHER` | 最近其他实体（非水晶、非目标）的距离场，多个实体逐格取 max |
| 9 | `CH_REACH` | 0/1 放置距离图：self 眼睛到格中心 ≤ 4.5。可以精确重算，直接作为输入 |
| 10 | `CH_CRYSTAL` | 最近末影水晶的距离场，敲击侧的主要输入 |

距离场定义，所有实体通道共用，SIGMA = 2.0：

```
d = 体素中心到实体碰撞箱表面的欧氏距离（箱内为 0）
value = exp(-d / 2)
```

体素中心 = `origin + (cx, y, cz) + (0.5, 0.5, 0.5)`。用连续距离场而不是 0/1 栅格，是因为被打分的位置在格子之间，二值表示会丢掉亚格信息。Java 侧用双精度构建，fp16 舍入由 golden 容差覆盖。

### aux：14 维 fp32，顺序固定

| 下标 | 值 | 归一化 |
|---|---|---|
| 0-2 | 目标速度 (x, y, z) | ÷ 0.35 |
| 3-5 | 自身速度 (x, y, z) | ÷ 0.35 |
| 6 | `aim_steps` | ÷ 5 |
| 7 | 目标护甲 | ÷ 20 |
| 8 | 目标护甲韧性 | ÷ 20 |
| 9 | 目标保护等级 | ÷ 20 |
| 10 | 目标血量 | ÷ 20 |
| 11 | 自身护甲 | ÷ 20 |
| 12 | 自身保护等级 | ÷ 20 |
| 13 | `difficulty` | ÷ 3 |

速度的恢复方式：`tvel/svel` 是每 tick 速度，从预测列表反推，`(last_pred - pos) / (aim_steps * stepTicks)`。数据帧按整数 tick 写预测，游戏内感知按半 tick 写，两者必须恢复出同一个每 tick 速度。没有预测时速度为零。

aux 不包含自身血量，也不包含战斗历史状态（无敌帧、爆炸时刻、水晶存量等）。

### 坐标约定

- crop 原点 = 目标脚格 + (-6, -3, -6)；网格下标 x∈[0,13)、y∈[0,7)、z∈[0,13)。
- 展平顺序：通道内 (y, z, x) 行主序，张量整体通道主序。
- Java 侧线性索引 `idx(x, y, z) = y * 13 * 13 + z * 13 + x`。
- 世界格 → 格中心是 (x+0.5, y+0.5, z+0.5)；伤害爆心用格底面中心 (x+0.5, y, z+0.5)，两者不同，后者是原版语义。

## 三、标签

- **dmg**（放置与敲击共用）：训练时在监督格集 `sup = place ∪ breakable` 上逐帧归一化，即除以帧内最大伤害；帧内最大伤害 < 0.5 半心的帧目标置零并跳过伤害损失。所以模型学的是伤害场的形状和排序，输出值是分数不是半心，部署时也不能当伤害用，详见 [reproducing.md](reproducing.md)。
- **valid**：可放性的二分类监督，正例加权；部署时不参与决策。
- **sup / has_dmg**：监督掩码由 place 和 breakable 合成；`has_dmg` 标记本帧是否有有效的归一化目标。
- **breakable**：敲击侧候选格，在 dmg 损失里与 placeable 同权。
- 数据增广：旋转/镜像同时作用于 x、标签和 aux 的速度列。漏掉速度列会让网络学到"地形会转、惯性不会转"的错误关联，`train.py` 启动时自检这一点。

## 四、golden 回归测试

特征和推理各有两份实现——特征是 Python 和 Java，推理是 ORT 和 DLL。golden 测试保证实现之间结果一致。金样本 `golden.bin` 约 620 KB，随仓库放在 `pocmod/src/test/resources/`，由 `ml/dump_golden.py` 生成，格式：

```
magic "FCG1" | u32 frameCount
每帧:
  u32 payloadLen | payload（.ncds 帧原文字节）
  f32 x[11 × 1183] | f32 aux[14] | f32 dmg[1183] | f32 valid[1183]
```

选帧规则：确定性均匀抽 6 帧含水晶帧 + 2 帧空场帧，x 按训练时的 fp16 舍入逐位重建。payload 包含 FeatureBuilder 需要的全部信息，Java 侧从 payload 重建特征。

两个测试，容差都 ≤ **1e-3**：

1. `FeatureGoldenTest`：python features.py 和 Java FeatureBuilder 在同样 8 帧上逐元素一致。
2. `InferGoldenTest`：python ORT 和 crystalnet.dll 内嵌 ORT 的 dmg/valid 逐元素一致。

两个测试连起来：python 特征 == Java 特征，python ORT == DLL，所以 DLL 在 Java 特征上跑出来的就是那个模型。`gradlew :pocmod:test` 一条命令验证，任何一处不一致都会失败。
