# crystalnet.dll

pocmod 的推理后端：一个内嵌 ONNX Runtime 1.31 的会话 + JNI 边界。ORT 在 `nativeInit` 时按 JVM 传入的路径动态加载（`LoadLibraryW` + `GetProcAddress("OrtGetApiBase")`），`onnxruntime.dll` 不进 git，构建只依赖 vendored 的 C API 头。DLL 用静态 CRT（/MT）构建，原因见下文"运行库要求"。

## 目录结构

```
native/
  xmake.lua                        # msvc /MT /utf-8，target crystalnet → build/bin/
  src/wire.h                       # 线协议常量（版本 1；11×1183 f32 x / 14 aux / 1183×2 out）
  src/infer.h + infer.cpp          # ORT 动态加载 + 固定 shape 会话 + 预分配 IO 张量
  src/bridge.cpp                   # JNI 四导出，符号名镜像 CrystalNetBridge.java
  third_party/onnxruntime/
    include/onnxruntime_c_api.h    # vendored（一次性获取，见下）
    lib/onnxruntime.dll            # 下载件，gitignore
    download/                      # 下载缓存，gitignore
```

## 构建

前置：[xmake](https://xmake.io)、MSVC（xmake 自动探测）、一个 JDK 21（提供 `jni.h`）。

```bash
# 1) JDK 头文件：设好 JAVA_HOME，或显式指定
xmake f -m release --jdk_root=<JDK21 安装目录>

# 2) ONNX Runtime 1.31.1（一次性；两个文件都不进 git）
#    头文件（小），从 NuGet 包取：
#      https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.31.1
#      → 解压包内 build/native/include/onnxruntime_c_api.h
#        到 third_party/onnxruntime/include/
#    运行时（约 19MB），二选一：
#      a. 同一 NuGet 包内 runtimes/win-x64/native/onnxruntime.dll
#      b. Python 侧 onnxruntime 包自带的同版本 DLL
#         （ml 的 deploy 依赖组锁定的版本与之一致）
#         → 拷 site-packages/onnxruntime/capi/onnxruntime.dll
#           到 third_party/onnxruntime/lib/
cd native && xmake
```

产物 `native/build/bin/crystalnet.dll`。`CrystalNetBridge.locateDll()` 从 pocmod 测试与 runClient 两种工作目录都能解析到它；ORT/模型路径同理，`-Dfcpoc.ort.path` / `-Dfcpoc.model.path` 可覆盖。

`pocmod` 打 fat jar 时会把 crystalnet.dll、onnxruntime.dll、`ml/runs/w24v2gpu/model.onnx` 一并嵌入（缺哪个就跳过哪个）；缺件时 jar 仍可构建，只是没有原生推理，开启时桥接层会报告缺失项。

## 线协议（WIRE_VERSION 1）

协议常量在 `src/wire.h`，Java 侧镜像在 `CrystalNetBridge.java`。版本不符直接报错；改协议需要两侧同步修改。

| 导出 | 语义 |
|---|---|
| `nativeVersion()` | 返回 1；Java 侧不符即报错 |
| `nativeInit(ortPath, modelPath)` | 动态加载 ORT 并建会话。错误码：1 ORT 加载失败 / 2 API 版本不符 / 3 模型不可读 / 4 会话创建失败 |
| `nativeInfer(x, aux, dmg, valid)` | 直连 ByteBuffer，小端 f32：x 52052 B（11 通道 × 1183 格，(y,z,x) 行主序）、aux 56 B（14 维）、输出 dmg/valid 各 4732 B（1183 格） |
| `nativeClose()` | 释放会话与张量 |

会话约束：

- 输入 shape 固定 `(1, 11, 7, 13, 13)`，输出 `(1, 7, 13, 13)` × 2。全部 IO 张量初始化时一次性预分配，推理路径上零分配。
- `intra_op = inter_op = 1`：单线程推理，避免与 Minecraft 渲染线程争核。
- Run 失败时输出清零，避免复用过期结果。

## 运行库要求（MSVC CRT）

ONNX Runtime 1.31 官方 Windows 发行版是 /MD 构建的：进程内它会绑定宿主 JVM 已经加载的那份 `msvcp140.dll`，而不是自己带的那份。实测结果：

- corretto 21.0.4 自带 msvcp140 14.29，加载后首次调用即 `EXCEPTION_ACCESS_VIOLATION`。
- Temurin 21（msvcp140 14.40）正常。

本仓库的默认配置已经避开这个问题：`pocmod` 把 Test 与 runClient 的 JVM 固定为 Temurin（ADOPTIUM vendor），`settings.gradle` 配了 foojay resolver，本机没有 Temurin 21 时 Gradle 自动下载。正常装一个 Temurin 21 即可。若用其他方式（自定义 launcher、外部进程）加载本 DLL，请确认宿主 JVM 的 msvcp140 版本 ≥ 14.40。

## 测试

```bash
gradlew :pocmod:test
```

两个 golden 测试验证整条链，容差均 ≤ 1e-3：`FeatureGoldenTest`（python features.py == Java FeatureBuilder）和 `InferGoldenTest`（python ORT == crystalnet.dll）。金样本 `golden.bin` 随仓库走，生成方式见 [ml/README.md](../ml/README.md)。两个测试连起来意味着：DLL 在 Java 特征上跑出来的就是那个模型。

游戏内实测数据见 [docs/reproducing.md](../docs/reproducing.md)。
