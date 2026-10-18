# ByteTrack 交通视频分析

本项目是一个面向固定机位道路视频的离线分析工程，基于 [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack) 的公开实现进行适配与工程化。仓库固定在上游提交 `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`，保留 MIT 许可证。上游 README 已归档至 `docs/UPSTREAM_README.md`。

项目目标不是训练新模型，而是把 ByteTrack 的多目标跟踪能力落到可复现的道路视频分析流程中：上传视频、绘制 ROI 与计数线、运行 A/B/C 三种处理模式、查看跟踪与计数结果、导出 CSV/JSON，并支持人工复核。

## 适用场景

- 固定机位道路视频的离线分析
- 车辆与骑行者等目标的跟踪、方向判断与计数
- 算法适配、可复现对比、部署验证与测试证据展示
- 作为独立工程练习，不涉及实际铁路信号控制，也不代表任何机构立场

## 核心能力

- 5 类 COCO 目标：汽车 `2`、公交 `5`、卡车 `7`、摩托 `3`、自行车 `1`
- 多边形 ROI 与一条有限双向计数线，锚点位于目标框底部中心
- 方向判定基于几何线方向，不表示地理上的进入/离开
- 死区为 `0.005 * height`，用于抑制线附近抖动
- 按 ID 维护方向生命周期，避免重复计数
- 帧索引从 0 开始
- A/B/C 三种模式共享同一 YOLOX-S COCO 预训练权重缓存，不重新训练

## A/B/C 模式说明

- A：上游演示语义，`frame_rate=30`，空帧跳过，宽高过滤系数 `1.6`
- B：使用实际 FPS，空帧更新，移除宽高过滤
- C：在最近 30 次检测结果上做加权类别投票

A 到 B 的计数方向变化来自几何线两侧的正负号变化，而不是地理意义上的进入/离开。方向标签读取自几何线方向。

## 快速开始

### 1. 环境准备

已验证环境：Windows、Python 3.12、PyTorch 2.5.1 + cu121、RTX 3070 Laptop 8GB。Python 3.10–3.12 为预期支持范围。

CPU 与 CUDA 安装方式不同，请按官方索引选择：

- CPU：`https://download.pytorch.org/whl/cpu`
- CUDA 12.1：`https://download.pytorch.org/whl/cu121`

Windows 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_weights.py
```

Linux 示例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_weights.py
```

`requirements-dev.txt` 已包含运行时依赖。权重下载脚本会校验 SHA。默认权重路径为 `weights/yolox_s.pth`，该文件已被忽略，不提交到仓库。

YOLOX-S 权重 SHA256：

```text
f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30
```

官方权重发布页：

```text
https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

FFmpeg 需要位于 `PATH` 中。

### 2. 启动 Web 界面

Windows：

```powershell
.\start.ps1
```

Linux：

```bash
python -m traffic.server
```

浏览器访问：

```text
http://127.0.0.1:8765
```

界面支持上传最大 512MB、最长 300 秒、最高 4K 的视频；可绘制 ROI 与计数线；可冻结实验配置；A/C 同步播放，B 可选；支持帧步进；人工真值独立于预测结果；支持草稿与显式复核；可导出 CSV/JSON 视频、检测缓存等结果。

### 3. 命令行运行

```bash
python -m traffic.cli video input.mp4 --output results/demo
```

复用缓存：

```bash
python -m traffic.cli video input.mp4 --output results/replay --cache results/demo/detections.jsonl
```

配置可从字典加载。最小 JSON 示例只需 ROI 与计数线，其余字段可选。

## 测试

```bash
python -m pytest tests -q
```

最近一次验证：2026-10-18，Windows、Python 3.12、PyTorch 2.5.1 + cu121、RTX 3070 Laptop 8GB，88 项测试全部通过。测试分布：核心 40、视频 6、评估 14、API 18、处理器 10。存在一条 Starlette/AnyIO 弃用警告，不影响功能测试。

Linux CI 已提供 Ubuntu CPU 工作流，但执行结果尚未在 GitHub 实际运行中验证。

## 真实 GPU Web 任务记录

最近一次真实 GPU Web 任务：`ddc34ebc5a1a42ba9a1a8e15bf40f416`，devbridge 前 10 秒，1920x1080，fps 30，共 300 帧。

| 指标 | A | B | C |
| --- | ---: | ---: | ---: |
| 检测器耗时（秒） | 7.4332621 | 7.4332621 | 7.4332621 |
| 跟踪耗时（秒） | 0.2475573 | 0.2650990 | 0.2644154 |
| 含渲染模式耗时（秒） | 4.21535 | 4.17071 | 4.27831 |
| 编码耗时（秒） | 2.04722 | 2.06905 | 2.67502 |
| 总耗时（秒） | 31.84795 | 31.84795 | 31.84795 |

该任务仅用于冒烟验证：夜间桥面、起始画面失焦，没有正式真值，也不覆盖 5 类目标。不要据此声称精度提升或质量结论。观测计数为 0 不能作为质量证明。A/B/C 源视频均为 H264、fps 30、300 帧，共享缓存哈希，报告状态为 `pending_human_review`。

性能表保留在 `docs/VALIDATION.md`，不在 README 中作为头条 FPS。纯跟踪器 FPS 不等同于端到端吞吐。

## 浏览器验收记录

已实际验证：中文界面与标题加载正常；可从原生文件选择器上传真实视频；可选择元数据与帧；可运行真实 GPU 任务并完成到进度 100；播放器下一帧时间 A=C=0.033333；媒体就绪状态为 4，无错误；复核页可加载 5 类目标，空真值独立；源步进帧为 1；空草稿可保存，`reviewed=false`，并绑定源、配置与任务哈希。

历史 404 已通过补充 `getHealth` 修复；修复后的当前流程未再出现额外错误。桌面 1512x1040 初始布局已修正为顶部标题加 290 侧栏；复核页溢出问题已发现并应用 CSS 修复，待重新检查。移动端 390 视口尚未测试，CSV 浏览器下载尚未测试，不应声称已验证。人工真值将由用户后续复核。

## 发布状态

当前为工程预览版，可发布代码与 PR，但不应打完整 `v0.1.0` 标签，也不应声称已完成。正式发布门槛：在独立相机开发/评估集上，每类至少 5 次清晰穿越；冻结参数后再进入留出集；不得在留出集上调参；需人工最终复核。

路线图中，RTSP 为 v1 之后的可选项；ReID、训练与速度优化不在当前范围内。

## 数据与许可

- 数据归属与链接：`docs/data-manifest.json`
- Commons 桥面作者 Editor，CC BY 3.0
- 原始与派生视频不提交到仓库
- YOLOX 许可证为 Apache 2.0，官方链接：`https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE`
- 权重发布受上游治理，不提交权重文件
- 上游项目：`https://github.com/FoundationVision/ByteTrack`
- 论文与 MIT 声明见上游仓库

## 说明

本项目为独立工程练习，与 Kedacom 无关联，不涉及实际铁路信号控制。
