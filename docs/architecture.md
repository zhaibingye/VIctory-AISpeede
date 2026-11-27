# 项目架构

第三代统一内存的职责、设计约束、代码入口及两阶段计划，见 [VNME 第三代架构与开发路线](vnme.md)。

```
 用户文件（仓库外）                       VIctory
 ────────────────────                    ──────────────────────────────────
 游戏资源 ─────────────────────────────► victory-pipeline + victory-game
                                         （victory-formats、victory-shader）
                                                   │
                                                   ▼
                                         cache_dir：转换结果
 游戏程序 ─────► victory-loader ────────► 游戏代码（原生 x86-64）
                准备、映射、绑定           │ 导入
                                          ▼
                                         victory-hle ── 宿主 Windows API
                                          │ D3D12.X
                                          ▼
                                         victory-gpu ──► 宿主 D3D12
                                          │ 创建管线时翻译着色器
                                          ▼
                                         victory-shader ◄──► cache_dir
```

游戏创建管线时，VIctory 翻译它传入的着色器字节，不区分这些字节来自游戏程序、归档还是内存。
`victory translate --cache` 从文件预先转换着色器，用同样的键填充同一份缓存。
再加 `--pipelines` 会根据引擎传入的创建流，为着色器包预建宿主管线并写入管线库。
`victory-game` 生成创建流，`victory-gpu` 按运行时的方式读取，使游戏启动后能按自身使用的名称查找这些管线。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `victory-core` | 配置、资料存放检查和日志 |
| `victory-pe` | 只读分析 PE32+ 和导入库 |
| `victory-loader` | 准备、映射、绑定并启动游戏程序 |
| `victory-hle` | 实现主机 API，管理允许直通的宿主 API |
| `victory-gpu` | 将 D3D12.X 对象、命令列表和 GPU 内存操作映射到宿主 D3D12 |
| `victory-memory` | 来宾页与物理身份、内存预算、驻留、一致性及队列时间线 |
| `victory-shader` | 将 Xbox 着色器容器里的程序翻译为宿主着色器 |
| `victory-formats` | GPU 纹理布局（大小、分块和解块）、BCPack 和音频编解码 |
| `victory-pipeline` | 通用转换接口与按内容寻址的缓存 |
| `victory-game` | 《侠盗猎车手 VI》专用格式、构建标识和钩子 |
| `apps/victory` | 命令行入口 |
| `xtask` | 资料存放规则检查和 Git 钩子 |
| `tests/fixtures/hle-probe` | 项目自己编写的替代游戏程序，用来测试加载器 |

只有 `victory-game` 负责单款游戏特有的逻辑；其余模块针对平台设计，可用其他 Xbox Series 游戏验证。这不代表已验证其他游戏兼容性。

## 导入解析

对游戏导入的每一个 `dll!function`，按以下顺序查找：

1. `victory-hle::modules` 中的实现优先。
2. 如果 `HostPolicy` 允许该 DLL 直通，则使用 Windows 自己的函数。
3. 如果前两项都不处理该 DLL，查找游戏程序同目录中自带的 DLL。加载器按与游戏程序相同的方式映射它，再查找导出。
4. 生成一个在调用时报告导入名称的占位函数。`--unimplemented trap` 会触发调试断点并终止，`--unimplemented zero` 只记录一次并返回 0。

对 `xgameruntime.dll` 等由本项目实现的游戏主机专用 DLL，`LoadLibrary*` 返回虚拟句柄，`GetProcAddress` 从本项目实现中查找函数。
对游戏自带的 DLL，`LoadLibrary*` 返回已经映射的映像，必要时先映射。
按名称加载与解析静态导入遵循同一套 `HostPolicy`；对直通句柄，`GetProcAddress` 也会采用导入绑定原本会选中的替代实现。

`QueryApiImpl` 按提供方与接口 ID 返回运行时对象，涉及的功能见 `victory-hle::modules::xgameruntime`。网络相关接口按离线主机响应。

游戏主机使用盘符访问存储。`victory-hle::modules::files` 绑定游戏从 kernel32 和 C/C++ 运行库导入的路径接口，把虚拟盘符下的路径转到宿主目录。持久化本地存储对应 `VICTORY_DATA_DIR` 下的目录；其他路径原样交给宿主。游戏存档也保存在 `VICTORY_DATA_DIR` 指定的位置。

游戏程序原本从槽位 0 访问静态线程局部存储（TLS）。按 DLL 映射后，槽位会变化，加载器会调整这些读取（`victory-loader::tls`）。
`victory run --execute` 会安装崩溃报告器，以 `module+rva` 标出故障位置并回溯出错线程（`victory-loader::crash`）。

`victory census` 不执行游戏，也不映射游戏自带的 DLL，只报告导入分类。`victory run` 不加 `--execute` 时会映射这些 DLL，但不启动游戏。

## GPU 内存与提交

[VNME 第三代架构与开发路线](vnme.md)记录物理页、别名、资源驻留、提交和同步的现行设计与验收边界。
[旧统一内存文档](unified-memory-engine.md)仅供回查演进历史。
开发中的 `paged-title` 路径只接收能按明确语义交给宿主 GPU 的命令。未覆盖的命令会报告拒绝原因，不会跳过绘制或伪造同步完成。静态编译通过不等于游戏已能进入主菜单。

## 着色器翻译

`victory_shader::ir::build` 把机器指令逐条转为静态单赋值（SSA）形式。
`victory_shader::structure` 随后消去硬件执行掩码 EXEC：将掩码改写为每个通道的布尔条件，将针对整个波前的掩码判断变成逐通道分支；仅在指令可能被 EXEC 禁用时，为向量指令保留条件保护。

当 `victory_shader::range::build` 能从逐通道形式证明寄存器相对寻址指令（`v_movrel*`）的索引范围时，会重跑上述两步，使指令只引用真正可能触及的寄存器。
`victory_shader::resource` 将描述符和标量内存地址追溯到对应的根参数（描述符表、堆、根描述符或常量）。

对循环中通过 `v_readfirstlane_b32` 读取的描述符，翻译器会沿通道实际经过的控制流，追溯到各通道原来的读取或选择操作，例如 `v_cndmask_b32` 或两个分支分别设置的值。这样可以用非一致描述符堆索引一次完成访问。
`victory translate` 对找到的着色器容器和 SGD5 文件执行这些转换，并列出尚不支持的情况。
