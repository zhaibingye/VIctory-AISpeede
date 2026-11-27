# VNME 项目记忆

## 入口与约束

- [`docs/vnme.md`](docs/vnme.md) 是现行架构与验收入口；[`docs/unified-memory-engine.md`](docs/unified-memory-engine.md) 仅供回查历史。来宾命令由 `crates/victory-gpu/src/list.rs` 录制，在 `page_queue.rs` 编码、提交和退休；`page_draw.rs` 做附件读写准入，`page_resource.rs` 按宿主对象和子资源记内容权威。`resource.rs` 管宿主对象共享与换代，`title_backing.rs` 和 `victory-memory/src/physical.rs` 管捕获的物理后备。
- 同一对象的权威按本批 `pending` → 同队列在途窗口的 `SubmittedTail` → 已完成版本查询。单 mip 单 slice 的颜色纹理用整资源键，多子资源用精确键；不跨对象或跨 mip 继承。GPU Fence 真正观察完成且必要的字节合并成功后才发布完成版本。
- 以失败证据而不是纹理尺寸做判断：覆盖整个目标的视口和裁剪不证明 draw 实际写全；转换屏障本身不生产像素。弱前代、相同 VA、当前查找代和同物理页都是不同层面的线索，单独都不证明分配代、布局或最新像素归属。

## 当前实机阻塞

- 最近实机包 `94861f1` 通过开场 XMem 准入，按 A 后仍在 draw 15 编码时首次拒绝；不是新的可玩基线。`13eee34` 在 6 GB 显存 Windows 机上触发 `draw` 索引 15 的 `MissingTextureContent`：slot 0 / subresource 0，格式 29，3840×2160×1，1 mip，`reads_old_color=true`。新证据是未记录同键前代、没有已识别的兼容字节来源；失败前缀无同对象或虚拟范围重叠的原生写入。索引 14 增强纹理屏障未声明弃旧。这不排除更早的提交、不同 VA 的物理别名或字节来源。
- `b40a245` 手动按 A 复现同一首错。`[VNME_PREFIX_COPY_INVENTORY]` 中来宾字节复制、显式纹理上传和纹理复制均为零；只涵盖**失败且未提交**批次的已编码前缀。`[VNME_NATIVE_WRITE_HISTORY]` 在拒绝点识别到目标 510 个物理页；保留的 128 个 Fence 内，256 个原生写入候选均归类为默认私有堆，未发现同对象、VA 重叠或共页；另有 600 个 Fence 的候选已淘汰。两次定时模拟 A 未触发首错，不能算通过；复现以可见继续提示时的手动输入为准。
- `3a63016` 手动复现同一首错。目标 510 页均为 XMem section，布局线索为 `SW_64KB_R_X`（27）；2,248 个原生写入 Fence 压缩为两组，未淘汰，成员均是默认私有堆候选；`[VNME_BYTE_WRITE_HISTORY]` 保留 10 个已完成的分页 GPU 写回页，目标命中为零，无跳过或淘汰。未观测的写入路径仍可能存在。
- `[VNME_DSTORAGE_WRITE_HISTORY]` 新增 CPU 侧证据：物理分段的身份取自写侧保活的 `Slice`，成功只记实际写入长度，原地失败仅记可能写过，裸 VA 只计未归属次数。与原生、GPU 字节历史共用一份目标快照；不回查历史 VA，不读取游戏内容。其他 CPU 写入、布局和当前权威仍须独立证明。`R_X` 已有分配尺寸计算，但生产路径没有适用于该目标的逐 texel 寻址证明，不能当作线性数据。

- `267f353` 仍复现同一首错；DirectStorage 未记录直接物理页写入，却观测到 9,896 次未归属的裸 VA 写入。这不是排除 DirectStorage 的证据。后续 `[VNME_DSTORAGE_VA_WRITE_HISTORY]` 将写前/写后目标页身份与 VA 范围独立比较：映射快照有采样限额，相同端点不证明中途未重绑。`[VNME_PREFIX_BYTE_INVENTORY]` 对照首错前已编码的 GPU 字节写入计划，不把未提交的命令说成已写入。

- `20d0ad1` 自行等开场帧计数稳定并确认游戏窗口在前台后发一次 Enter，复现同一首错。DirectStorage 的 9,894 次裸地址写入没有候选 VA 重叠，保留的 1,655 个 section 候选页也无命中；360 次采样不完整，不能排除盲区。未提交前缀声明 10 个脏页，也没有命中目标。下一步优先追目标创建、XMem 绑定/重绑与首次 RTV/SRV 用途，查清是否漏掉应有的可证明初始化；不凭未匹配推断为零初始化。

- `df5dcfd` 的 `[VNME_RTV_ORIGIN]` 确认为地址式创建，初始增强布局码 6，优化清除值存在，宿主无已知前代；`[VNME_XMEM_SECTION_HISTORY]` 的 510 个活跃页均匹配同一新绑定的 section，无相关解绑记录。`0x470` 在失败 draw 前同录制调用过一次，但 ABI 和语义未证实，不读取未知参数。桌面 D3D12 的布局枚举只供对照，不能把优化清除值解释为写像素。
- 无前台时可在诊断构建使用连续重叠的 `VICTORY_KEYS` 定时区间合成一次 A 长按；本地在输入轮询开始后 65–66.1 秒成功触发同一首错。未发出输入或仅停在开始画面不算实机通过。

- `7174c48` 证实实际 PSO 的 RGB/Alpha 都是 `ADD(SRC_ALPHA, INV_SRC_ALPHA)`，不是可以删掉的旧色依赖。`9e65b0d` 的一次性只读探针完整观测到目标 510 个 CPU 页均为零（无内容或指纹落盘），仅代表 CPU 当前读值。`e61c206` 显示前缀 8/10 两个颜色清除及 12/13 两个绘制都写别的默认私有堆对象（格式 28，而目标为 29），6 的 compute 输出不同且失败时物理页交集为零。不能按尺寸移交。下一步应从可信物理节的初始字节、物理写入时序和可验证的数据平面转换构建可独立回归的字节兼容路径，不从零观测直接推导清屏。

## 开场 XMem 的宿主 RAM 准入

- 原生 Windows `paged-title` 用 `SEC_RESERVE` 创建 section。`DeferredRam` 按 **section 内页号**预扣净新增 64 KiB 页；宿主提交失败退回新增额度，解绑单个 VA 不释放仍存活的 section。`xmem::tests::deferred_native_` 用真实重叠视图、只读视图提交失败、同 VA 换节后重试验证这些边界。拒绝日志的 `requested_mib` 是整段长度，不是净新增字节；不能据此断言重复计费。
- `xmem::tests::native_reserved_section_first_commit_is_zero_but_pinned_remap_keeps_bytes` 在真实 Windows section 上区分了三个时刻：无写者的新节首提交为零；`write_physical` 可在首次来宾映射之前提交并写入；提交固定页之后，尚存的 `page_window` 仍能写旧节，即使来宾已在同一 VA 绑定新节。`read_backing_runs` 能读回旧节的**后写值**，不是 pin 时的前像。首次绑定、固定身份、冻结字节版本不能互换；这些对照均不证明实机目标发生过相同写入。
- 启动时 `prepare` 持有原文件的完整补丁副本；`load` 返回的 `LoadedImage` 已有独立宿主映射和解析信息，无需把原始或准备缓冲保留到 `main.join()`。释放它们不释放 XMem 租约，也不改变预算公式；是否还有足够 RAM 须按当次采样验证。

## 共享字节读取的生产门禁

- `page_resource.rs` 的 `record_write_at_with` 必须将“可能写共享后备”和“可发布整面原生颜色版本”分开。弃旧后只清半面是实际写入，但整面仍为 `Discarded`。`page_queue.rs` 在原生入队前检查本批及此前提交的未分类原生写；拒绝是保守冲突裁决，不是相交证明，更不是旧色生产者。回归 `page_queue::discarded_partial_color_clear_blocks_shared_texture_read` 覆盖本批反例。

- 共享体纹理 UAV 虽是输出，`ShaderTextureUav::prepare` 仍要从固定页预填充整面，以保留着色器未写的 texel。它与共享 SRV 同走 `SharedTextureBytes::gather_into`：逐有效区间验证独立后备，登记共享读取，交给提交门禁检查未分类原生写者。`verify_shared_volume` 只检查本条派发的视图与范围，不能代替这两项。`page_uav::prefill_tests` 有同页双 VA 不同宿主对象、后续 W 分片来源未证的反例。这条规则不为 R_X 颜色目标提供旧色。

- `TextureInput::Linear` 也可由地址式资源的 SRV 选中，不能从 `Linear` 推断后备私有。带外部别名的线性采样逐行验证有效字节（不含 pitch 填充），与分块 SRV/UAV 共用读取登记。`page_queue::linear_read_tests` 走描述符选择和实际提交门禁：不同宿主对象、不同 VA 的清除与读取可以指向同一固定页；仅查读者自身原生版本会漏冲突。无写者且无历史冲突时仍可读取。

- 已完成 Fence 的物理页写回也不能按整页认领：`title_backing::merge_page` 只写提交前像与 GPU 回读的差异及显式覆盖的并集，即使两者分成很多段。整页读改写会回写未归 GPU 的间隙，既可能丢失并行 CPU 更新，也可能因间隙不可写而失败。`title_backing::writeback_tests` 用夹在可写片段之间的只读宿主页区分两条路径。只保证不触及间隙，不裁决 CPU/GPU 对同一字节的冲突。

- 单次 `GuestCopy` 的源和目的重叠可经暂存保留源字节，但暂存不能证明**目的自身**的映射是单射。`copy_plan::GuestCopy::bind` 用提交固定页和页内半开区间检查；不同目的 VA 在同一物理页写到不同字节是合法的，若争写同一字节则在编码时拒绝整批。不同命令的有序覆盖不是这个反例。见 `copy_plan::alias_tests` 和 `page_queue::paged_buffer_copy_rejects_self_aliased_destination_before_submitting_prefix`。

## 已有的受限字节转换

- HLE `modules/xg.rs::layout` 输出的是 `victory_formats::texture::Texture::new` 建模的 data/depth/stencil plane；该模型没有 DCC 等元数据平面。GPU `resource::texture_layout` 使用同一构造器。因此默认 RTV 的 XG 单 plane、offset 0、与 GPU 计算大小相符，不是主平面无需元数据解释的独立证据。

- `victory_formats::texture::copy_uniform_rx_rgba8_blocks` 只处理独立证明为无压缩主平面的 2D `SW_64KB_R_X`、单层单 mip、单采样、4 字节 texel。按 GFX10 addrlib 的块索引，128×128 个 texel 占一块，块内 XOR 不改变块号。逐块验证全部四字节字相同（含填充）即可恢复紧密行；不同块不必同色。`uniform_rx` 集成测试有六个异色块及末块单字节反例。这不是通用逐像素寻址，未接入生产；510 页的 CPU 零观测不能替代平面描述、提交时的权威快照或同步证明。

- `copy_gfx10_1pipe_rx_rgba8` 可恢复非均匀块，但只实现公开 GFX10 非 RB+、单 pipe、256 B interleave、`pipeBankXor=0` 的上述主平面。在 `gfx10_rx` 夹具中，`Addr2ComputeSurfaceAddrFromCoord` 对同一幅 259×131 合成图的 texel (63,64) 在 1 和 2 pipe 下分别给出平面内字节偏移 `0x6a8c` 与 `0x6b8c`。布局码、行距、格式和大小不能选出方程。该函数不改变 `Layout::addressable()`，不用于判断游戏配置或旧色权威。

## 最小验证

- 使用仓库锁定的 Rust、Windows x64 MSVC、`--locked --features paged-title,diagnostics,blake3/pure`。新增诊断的定向测试过滤 `page_resource::native_history_tests`、`page_queue::copy_prefix_`；内容顺序的实像素测试为 `page_draw::paged_title_partial_color_clear_observes_submission_order`，仅相关改动时重跑。
- 构建按提交分目录，先核对程序的 SHA-256 和来源再实跑。不同程序或触发条件用独立日志；仅提炼脱敏证据，不保存游戏数据、原始地址或私人路径到仓库。尚未达到可玩验证，不以离线测试或菜单停留替代。
