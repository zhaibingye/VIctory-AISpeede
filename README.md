# VIctory

VIctory 在 Windows 上原生运行 Xbox Series 游戏。这个分支只支持 Series S；
运行所需的游戏文件由有权使用它们的用户自行提供。

游戏的 x86-64 指令在 VIctory 进程中直接执行。VIctory 提供运行所需的平台接口，
并将着色器、图形管线和资源转换成 PC 可使用的形式。能提前完成的转换可以预编译，
其余在首次使用时执行。详见[架构说明](docs/architecture.md)。

## 当前状态

**尚不能正常游玩。** 下表是 2026-11-24 在 Windows 11、AMD RX 6700 XT 上
记录的既有渲染路径结果，不是 VNME 开发分支的实机表现，也不是最低配置保证。

| 阶段 | 既有路径的结果 |
|---|---|
| 首次启动界面（无障碍、语言、设置和法律声明） | 可运行，约 30 帧/秒 |
| 开场视频 | 可播放，有声音 |
| 开始界面、用户配置和存档 | 可运行 |
| 加载游戏世界 | 当时约需 50 秒 |
| 第一段任务演出 | 可以显示街道、建筑、车辆、角色、标题和字幕；画面缺光照，部分远处几何和角色显示异常，单帧约需 1 至 2 秒 |

第三代内存引擎 VNME 仍在开发，使用 `paged-title` 功能显式开启。首个启动
诊断包只验证到了游戏初始化阶段，尚未确认能进入主菜单。该诊断包不是正式发布。

> 较早的版本进入游戏世界时曾导致 GPU 挂起，Windows 可能重置显示驱动，
> 短暂黑屏。遇到 GPU 错误可查看 `crashes.log`。诊断版可使用
> `VICTORY_D3D12_WARP=1` 对照软件渲染，速度显著低于实机硬件。

### 既有功能和已知限制

| 模块 | 已有功能 | 限制 |
|---|---|---|
| 加载器 | 在进程内运行游戏 EXE 和 DLL，将错误位置记入 `crashes.log` | |
| 运行时服务 | 单个离线用户、包管理、存档与文本输入界面；异常退出后保留已写入的存档 | 语音服务无声，文本界面及重启尚未实机确认 |
| 网络 | 默认离线，游戏不可访问网络 | |
| 文件与流式加载 | 读取归档和流式数据，并行解码 zlib 和 BCPack | 加载易受写入游戏内存的开销限制 |
| 输入 | 支持 XInput 手柄，也可用键盘模拟手柄 | 未支持扳机振动 |
| 音频 | 菜单、视频及加载声音；解码 XMA 与 Opus | 混响无声，Opus 输出尚未逐项对照 |
| 菜单渲染 | 通过宿主 D3D12 绘制、计算和呈现，只复制变动的游戏内存 | 发布构建在某些屏幕上帧间隔不稳定 |
| 世界渲染 | 支持网格管线、间接执行、着色器动态选取堆描述符、按地址读取游戏内存、计算写深度及磁盘管线缓存 | 存在光照和几何缺失；未翻译的着色器不绘制；既有路径的查询、深度边界、HDR 及缩放支持有限 |
| 着色器 | 转译为 DXIL，支持提前缓存或首次执行时转换；历史统计：世界网格 8,765/8,765，顶点与像素 12,550/12,592，计算 2,579/2,637 | 尚缺合并几何阶段（195 个程序）及约百个其他程序；融合乘加依宿主 GPU 执行，一些计算程序需要支持 64 线程波前的 GPU |
| 纹理 | 支持游戏使用的部分分块布局和纹理解排 | 渲染目标的元数据平面尚未支持 |

## 开始使用

- Windows 10 或 Windows 11 x64，CPU 支持 AVX2，GPU 支持 Direct3D 12 与
  Shader Model 6.6。世界渲染另需 GPU 支持网格着色器。
- 使用 [rustup](https://rustup.rs) 安装 Rust，也可运行
  `winget install --id Rustlang.Rustup -e`。仓库通过 `rust-toolchain.toml`
  固定工具链版本。
- 源码构建需 Visual Studio 或 Visual Studio Build Tools 的「使用 C++ 的桌面开发」工作负载。

游戏启动需要数 GiB 的 Windows 提交额度。可在「任务管理器 → 性能 → 内存」中查看
「已提交」。若 `crashes.log` 报 Windows 错误 1455，先关闭其他应用或增大分页文件，再重试。

```powershell
cargo xtask install-hooks                # 每次提交时检查资料存放规则
copy victory.example.toml victory.toml   # 填写仓库外的路径
cargo test --workspace                   # 开发者可选，不需要游戏文件
cargo run -p victory -- config           # 检查生效配置
```

构建并启动：

```powershell
cargo build --release -p victory
$game = 'D:\你的游戏目录'
$title = Join-Path $game 'game_scarlett_master.exe'
$victory = '.\target\release\victory.exe'
& $victory translate --cache --pipelines "$game\shaders.rpf" $title "$game\XboxBvhBuilder.dll" "$game\amd_fidelityfx_upscaler_xbox.dll"
& $victory run $title --execute
```

仍兼容 `--console series-s`。内存参数见[Series S 内存说明](docs/series-s-memory.md)。

`translate --cache` 可将扫描出的着色器提前写入磁盘缓存（历史测试约 1.3 GB）。
不做这一步时，着色器在首次使用时现场翻译。加上 `--pipelines` 会把
`shaders.rpf` 中的可用图形与计算管线也预建到宿主管线库。历史样本约
27,000 条管线、缓存约 4 MB，16 个线程约需 20 分钟。这些数字不是当前分支的实测。
管线库与创建它的 GPU 和驱动有关；中途停止后再次运行只补齐未完成项。

### 目录和画面

默认在 `victory.exe` 所在目录创建 `data` 和 `cache`。前者保存存档、设置和
`crashes.log`，后者保存可重建的缓存。环境变量和 `victory.toml` 中显式设置的
路径优先。旧文件不会自动搬动。可执行文件若位于源码仓库内，请通过
`VICTORY_CACHE_DIR` 和 `VICTORY_DATA_DIR` 或配置文件指定仓库外的路径。

Series S 的放大功能使用 AMD FidelityFX。若要在宿主 GPU 上使用 PC 版本，可安装
[FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)，
并令 `VICTORY_FIDELITYFX_DIR`（或 `[paths].fidelityfx_dir`）指向
`Kits/FidelityFX/signedbin`，即包含 `amd_fidelityfx_upscaler_dx12.dll` 的目录。
没有可用的 PC 版时，使用游戏自带版本；某些情况下可以用双线性缩放填充输出，
但不能替代 FidelityFX 的时域质量，也不会改变游戏的内部渲染分辨率。

窗口标题显示帧率。设置 `VICTORY_WINDOW_MAX=1920x1080` 可以限制顶层窗口尺寸；
比上限小的窗口保持原样，超过上限的等比缩小，同时受显示器工作区限制。
这不会改变主机版内部渲染分辨率或交换链缓冲区。

### 键盘与手柄

支持 XInput 手柄。未接手柄时，游戏窗口获得焦点后可用以下按键：

| 键盘 | 手柄 |
|---|---|
| W A S D / I J K L | 左 / 右摇杆 |
| 方向键 | 十字键 |
| Enter、Backspace、X、Y | A、B、X、Y |
| Q / E、Z / C | 肩键、扳机 |
| F / H | 按下摇杆 |
| Esc、Tab | Menu、View |

## 命令

```text
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm] [--effects]
victory translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
victory translate [paths...] --cache [--pipelines] [--jobs N]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [-- <title arguments>]
victory config
```

可用 `VICTORY_LOG=victory_loader=trace,info` 设置日志筛选。普通发布构建只会
输出 `info` 及更高级别；更细的日志需要诊断版。

若 Windows 终端中的中文显示为乱码，推荐设置 `$env:VICTORY_LOG_FILE = 'vnme-startup.log'`。
程序会把运行日志直接追加到 UTF-8 文件，不经过 PowerShell 的管道解码。该文件
不包含单独通过命令行标准输出打印的内容；文件位置也必须位于源码仓库外。

如果需要用 PowerShell 保存完整的命令行输出，先在同一窗口执行
`[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)`，
再运行带有 `2>&1 | Tee-Object -FilePath vnme-startup.log` 的命令。
`Tee-Object` 无法修复已经被错误解码的旧日志。

### Windows 着色器验证

着色器测试使用 `dxil.dll` 验证 Shader Model 6.6。较旧的 Windows SDK
可能报 `Unknown shader model 'ps_6_6'`。可以改用
[微软 DirectX Shader Compiler](https://github.com/microsoft/DirectXShaderCompiler/releases)
或较新的 Windows SDK 中的 x64 验证器：

```powershell
$env:VICTORY_DXIL = 'C:\tools\dxc\bin\x64\dxil.dll'
cargo test -p victory-shader --lib --locked -- --test-threads=1
```

未指定时，先寻找已安装 SDK 中版本最新的 `bin/<version>/x64/dxil.dll`，
找不到再搜索 DLL 路径。已转换的着色器在正常运行时不依赖此验证器。

## 诊断

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

下表中值为 `1` 的开关，在未设置或值为 `0` 时默认关闭。

| 变量 | 作用 |
|---|---|
| `VICTORY_LOG=<target>=debug,...` | 输出符合筛选条件的调试和跟踪日志 |
| `VICTORY_STACKS=20,60` | 在指定秒数记录各线程调用栈 |
| `VICTORY_WATCH=<rva>,...` | 记录指定游戏地址处 32 位数值的变化 |
| `VICTORY_POKE=<rva>:<byte>,...` | 每秒两次写入指定字节，用于诊断受控测试里的看门狗等 |
| `VICTORY_KEYS=30:Enter,35:X` | 在指定时刻模拟按键，不依赖窗口焦点 |
| `VICTORY_NO_HOST_PAD=1` | 忽略 XInput 手柄，使用键盘模拟手柄 |
| `VICTORY_CRASH_BREAKPOINTS=1` | 将首次遇到的断点也写入 `crashes.log`，包括可能被处理的断点 |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | 发生错误时导出指定内存范围到缓存目录 |
| `VICTORY_CAPTURE=<dir>` | 保存呈现帧；可以用 `VICTORY_CAPTURE_FRAMES=<list>` 指定帧号 |
| `VICTORY_DUMP_TARGETS=1` | 抓帧时保存所选帧的全部渲染目标 |
| `VICTORY_CAPTURE_WORLD=<list>` | 结合抓帧，从首个达到 500 次绘制的帧起，按相对帧号保存；也可导出渲染目标 |
| `VICTORY_DUMP_WRITERS=<hash>,...` | 结合抓帧，保存指定着色器所在管线首次运行时写入的纹理，并记录缓冲区变动 |
| `VICTORY_PEEK=<hex address>[:<bytes>],...` | 在所选帧的每次提交后记录指定字节、较小的渲染目标与缓冲区、描述符，以及计算写入的一维或三维纹理 |
| `VICTORY_D3D12_DEBUG=1` | 开启宿主 D3D12 调试层 |
| `VICTORY_D3D12_GBV=1` | 配合调试层开启 GPU 验证，帮助发现错误描述符，速度较慢 |
| `VICTORY_D3D12_DRED=1` | GPU 出错时记录相关分配和在途命令 |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | 忽略包含指定 16 位着色器哈希的管线的绘制和调度，包括间接命令 |
| `VICTORY_GPU_TIMING=1` | 将既有路径的命令列表分开提交并计时；配合 DRED 显示命令详情，单独开启时可能显示 `commands=0` |
| `VICTORY_VNME_TIMING=1` | 在启用 VNME 的构建中记录提交的 CPU 耗时、分页传输和受管内存，不是 GPU 帧时间 |
| `VICTORY_TRACE_SHADERS=<hash>,...` | 配合 DRED 记录指定着色器的绘制、调度和绑定，最多 500 项 |
| `VICTORY_SPARSE_BC1=0` | 关闭符合条件的只读 BC1 纹理的按需驻留 |
| `VICTORY_D3D12_WARP=1` | 使用 WARP 软件适配器，速度较慢 |
| `VICTORY_D3D12_ADAPTER=NVIDIA` | 按适配器名称或从零开始的 DXGI 索引选 GPU；WARP 模式忽略此设置 |
| `VICTORY_PIPELINE_LOG=<file>` | 记录管线的创建、首次使用、缓存命中和预建 |
| `VICTORY_PREBUILD_KEYS=<file>` | 仅列出预建候选管线的键、着色器、状态或跳过原因，不实际编译 |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | 将转换后着色器单线程的最大循环次数改为 `n`，使用独立缓存；可用于判断是否被长循环拖慢 |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | 录制本进程的声音 |
| `VICTORY_XMA_DUMP=<dir>` | 保存前 32 路 XMA 的上下文与输入 |

所有构建都会将未处理的异常、程序崩溃和宿主 GPU 错误附加到标题数据目录的
`crashes.log`。若 VNME 启动失败，请保留首次包含 `[VNME_SUBMISSION_REJECTED]`
或 `UnloweredList` 的日志及前后几行；若未出现，保留最早的一条报错。

旧版启动时的断点 `0x80000003` 可用诊断版与 `VICTORY_CRASH_BREAKPOINTS=1`
定位。若系统报告宿主 DLL 拒绝加载，控制台会记录调用栈。策略范围以外的宿主 DLL
被拒绝是预期行为，例如 `rpcrt4.dll`。对于 Series 1.0.2.1，缺少
`appdata0.rpf` 曾在窗口出现前触发断点 `+0x2237178`；`run --execute`
会先检查同一游戏目录的 RPF9 头。

如需对照软硬件渲染，可用 WARP 再运行一次，确认控制台确实选择了 WARP
软件适配器，并对比两次异常的模块、偏移、调用栈及前置日志。

## 许可证

GPL-3.0-or-later。以 [LICENSE](LICENSE) 中的原文为准。
