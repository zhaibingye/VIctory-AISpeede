# VNME 项目记忆

## 入口与约束

- [`docs/vnme.md`](docs/vnme.md) 是现行架构与验收入口；[`docs/unified-memory-engine.md`](docs/unified-memory-engine.md) 仅供回查历史。来宾命令由 `crates/victory-gpu/src/list.rs` 录制，在 `page_queue.rs` 编码、提交和退休；`page_draw.rs` 做附件读写准入，`page_resource.rs` 按宿主对象和子资源记内容权威。`resource.rs` 管宿主对象共享与换代，`title_backing.rs` 和 `victory-memory/src/physical.rs` 管捕获的物理后备。
- 同一对象的权威按本批 `pending` → 同队列在途窗口的 `SubmittedTail` → 已完成版本查询。单 mip 单 slice 的颜色纹理用整资源键，多子资源用精确键；不跨对象或跨 mip 继承。GPU Fence 真正观察完成且必要的字节合并成功后才发布完成版本。
- 以失败证据而不是纹理尺寸做判断：覆盖整个目标的视口和裁剪不证明 draw 实际写全；转换屏障本身不生产像素。弱前代、相同 VA、当前查找代和同物理页都是不同层面的线索，单独都不证明分配代、布局或最新像素归属。

## 当前实机阻塞

- 最近实机包 `64dd795` 已启用 DirectStorage 裸 VA 写窗口；与 `94861f1` 一样，通过开场 XMem 准入，按 A 后仍在 draw 15 编码时首次拒绝，未出现更早的上传快照冲突。目标的 510 张页匹配同一节；这一包未探测颜色字节。不是新的可玩基线。`13eee34` 在 6 GB 显存 Windows 机上触发 `draw` 索引 15 的 `MissingTextureContent`：slot 0 / subresource 0，格式 29，3840×2160×1，1 mip，`reads_old_color=true`。新证据是未记录同键前代、没有已识别的兼容字节来源；失败前缀无同对象或虚拟范围重叠的原生写入。索引 14 增强纹理屏障未声明弃旧。这不排除更早的提交、不同 VA 的物理别名或字节来源。
- `b40a245` 手动按 A 复现同一首错。`[VNME_PREFIX_COPY_INVENTORY]` 中来宾字节复制、显式纹理上传和纹理复制均为零；只涵盖**失败且未提交**批次的已编码前缀。`[VNME_NATIVE_WRITE_HISTORY]` 在拒绝点识别到目标 510 个物理页；保留的 128 个 Fence 内，256 个原生写入候选均归类为默认私有堆，未发现同对象、VA 重叠或共页；另有 600 个 Fence 的候选已淘汰。两次定时模拟 A 未触发首错，不能算通过；复现以可见继续提示时的手动输入为准。
- `3a63016` 手动复现同一首错。目标 510 页均为 XMem section，布局线索为 `SW_64KB_R_X`（27）；2,248 个原生写入 Fence 压缩为两组，未淘汰，成员均是默认私有堆候选；`[VNME_BYTE_WRITE_HISTORY]` 保留 10 个已完成的分页 GPU 写回页，目标命中为零，无跳过或淘汰。未观测的写入路径仍可能存在。
- `[VNME_DSTORAGE_WRITE_HISTORY]` 新增 CPU 侧证据：物理分段的身份取自写侧保活的 `Slice`，成功只记实际写入长度，原地失败仅记可能写过，裸 VA 只计未归属次数。与原生、GPU 字节历史共用一份目标快照；不回查历史 VA，不读取游戏内容。其他 CPU 写入、布局和当前权威仍须独立证明。`R_X` 已有分配尺寸计算，但生产路径没有适用于该目标的逐 texel 寻址证明，不能当作线性数据。

- `267f353` 仍复现同一首错；DirectStorage 未记录直接物理页写入，却观测到 9,896 次未归属的裸 VA 写入。这不是排除 DirectStorage 的证据。后续 `[VNME_DSTORAGE_VA_WRITE_HISTORY]` 将写前/写后目标页身份与 VA 范围独立比较：映射快照有采样限额，相同端点不证明中途未重绑。`[VNME_PREFIX_BYTE_INVENTORY]` 对照首错前已编码的 GPU 字节写入计划，不把未提交的命令说成已写入。

- `20d0ad1` 自行等开场帧计数稳定并确认游戏窗口在前台后发一次 Enter，复现同一首错。DirectStorage 的 9,894 次裸地址写入没有候选 VA 重叠，保留的 1,655 个 section 候选页也无命中；360 次采样不完整，不能排除盲区。未提交前缀声明 10 个脏页，也没有命中目标。下一步优先追目标创建、XMem 绑定/重绑与首次 RTV/SRV 用途，查清是否漏掉应有的可证明初始化；不凭未匹配推断为零初始化。

- `df5dcfd` 的 `[VNME_RTV_ORIGIN]` 确认为地址式创建，初始增强布局码 6，优化清除值存在，宿主无已知前代；`[VNME_XMEM_SECTION_HISTORY]` 的 510 个活跃页均匹配同一新绑定的 section，无相关解绑记录。`0x470` 在失败 draw 前同录制调用过一次，但 ABI 和语义未证实，不读取未知参数。桌面 D3D12 的布局枚举只供对照，不能把优化清除值解释为写像素。
- 无前台时可在诊断构建使用连续重叠的 `VICTORY_KEYS` 定时区间合成一次 A 长按；本地在输入轮询开始后 65–66.1 秒成功触发同一首错。未发出输入或仅停在开始画面不算实机通过。

- `7174c48` 证实实际 PSO 的 RGB/Alpha 都是 `ADD(SRC_ALPHA, INV_SRC_ALPHA)`，不是可以删掉的旧色依赖。`9e65b0d` 的一次性只读探针完整观测到目标 510 个 CPU 页均为零（无内容或指纹落盘），仅代表 CPU 当前读值。`e61c206` 显示前缀 8/10 两个颜色清除及 12/13 两个绘制都写别的默认私有堆对象（格式 28，而目标为 29），6 的 compute 输出不同且失败时物理页交集为零。不能按尺寸移交。下一步应从可信物理节的初始字节、物理写入时序和可验证的数据平面转换构建可独立回归的字节兼容路径，不从零观测直接推导清屏。

- `64dd795` 实跑归档中，891 个已完成 Fence 的 1,782 次原生写入候选均归类为默认图形堆；目标 510 个候选页属于同一 XMem section，记录内无同对象或 VA 重叠。这份历史没有完整的物理覆盖。独立核对表明，`CopyPageMappingsBatchX` 只复制 XMem 视图关系：登记页池不会导入分配，`map_alias` 只接收已登记的 section 源，目的必须能由 Windows 替换 placeholder。`xmem::tests::managed_xmem_alias_shares_section_not_live_private_memory` 在原生 Windows 证明同节两 VA 既被提交固定为同页又双向共享真实字节；活跃、64 KiB 对齐的默认图形堆没有 section 身份且不能当页池源，普通已提交私有目的也拒绝替换并保留旧字节。只排除已实现映射的直接共用字节，不排除间接复制、其他写入、漏译或游戏本应有而 VNME 尚未实现的语义。

## 开场 XMem 的宿主 RAM 准入

- 原生 Windows `paged-title` 用 `SEC_RESERVE` 创建 section。`DeferredRam` 按 **section 内页号**预扣净新增 64 KiB 页；宿主提交失败退回新增额度，解绑单个 VA 不释放仍存活的 section。`xmem::tests::deferred_native_` 用真实重叠视图、只读视图提交失败、同 VA 换节后重试验证这些边界。拒绝日志的 `requested_mib` 是整段长度，不是净新增字节；不能据此断言重复计费。
- `xmem::tests::native_reserved_section_first_commit_is_zero_but_pinned_remap_keeps_bytes` 在真实 Windows section 上区分了三个时刻：无写者的新节首提交为零；`write_physical` 可在首次来宾映射之前提交并写入；提交固定页之后，尚存的 `page_window` 仍能写旧节，即使来宾已在同一 VA 绑定新节。`read_backing_runs` 能读回旧节的**后写值**，不是 pin 时的前像。首次绑定、固定身份、冻结字节版本不能互换；这些对照均不证明实机目标发生过相同写入。
- 启动时 `prepare` 持有原文件的完整补丁副本；`load` 返回的 `LoadedImage` 已有独立宿主映射和解析信息，无需把原始或准备缓冲保留到 `main.join()`。释放它们不释放 XMem 租约，也不改变预算公式；是否还有足够 RAM 须按当次采样验证。

## 共享字节读取的生产门禁

- `CapturedUploads::stage` 在拷入第一张页之前，以 `physical::access::try_snapshot` 一次登记整批固定的 section 页，持有到最后一页拷完。`PageWindow` 和 `write_physical` 使用相同物理身份登记完整写窗口；写者等待已有快照退出，新快照遇到相交写者在入队前报 `PhysicalWriterDuringSnapshot`。同址换节不冲突；不能用逐页加锁代替整批保护，否则跨页写的两侧可被拷成两个版本。此协议不覆盖任意来宾 CPU 或原生 GPU 写，不证明 R_X 平面或 draw 15 的旧色。对照见 `xmem::snapshot_tests` 和 `page_queue::paged_copy_snapshot_rejects_physical_writer_before_enqueue`。

- `paged-title` 的 DirectStorage 裸 VA fill 在整个同步填充（含部分失败）期间持有受管标题视图共享门锁；`XMemMapPhysicalPages` 和 `map_alias` 在独占侧完成拆分、解绑和重映射。原因是只改一页也可能临时卸掉整个宿主视图。填充在门锁内解析**所有**相交 section 页，释放 `PHYSICAL` 才等上传快照；上传拷入不取视图门锁。未知/私有字节不获得 section 身份。多个 fill 可并行，但无关的受管重映射也可能等 I/O。本协议协调写入窗口，不发布像素版本，不能裁决拷入之后的同字节写回。

- `page_resource.rs` 的 `record_write_at_with` 必须将“可能写共享后备”和“可发布整面原生颜色版本”分开。弃旧后只清半面是实际写入，但整面仍为 `Discarded`。`page_queue.rs` 在原生入队前检查本批及此前提交的未分类原生写；拒绝是保守冲突裁决，不是相交证明，更不是旧色生产者。回归 `page_queue::discarded_partial_color_clear_blocks_shared_texture_read` 覆盖本批反例。

- 共享体纹理 UAV 虽是输出，`ShaderTextureUav::prepare` 仍要从固定页预填充整面，以保留着色器未写的 texel。它与共享 SRV 同走 `SharedTextureBytes::gather_into`：逐有效区间验证独立后备，登记共享读取，交给提交门禁检查未分类原生写者。`verify_shared_volume` 只检查本条派发的视图与范围，不能代替这两项。`page_uav::prefill_tests` 有同页双 VA 不同宿主对象、后续 W 分片来源未证的反例。这条规则不为 R_X 颜色目标提供旧色。

- `TextureInput::Linear` 也可由地址式资源的 SRV 选中，不能从 `Linear` 推断后备私有。带外部别名的线性采样逐行验证有效字节（不含 pitch 填充），与分块 SRV/UAV 共用读取登记。`page_queue::linear_read_tests` 走描述符选择和实际提交门禁：不同宿主对象、不同 VA 的清除与读取可以指向同一固定页；仅查读者自身原生版本会漏冲突。无写者且无历史冲突时仍可读取。

- 已完成 Fence 的物理页写回也不能按整页认领：`title_backing::merge_page` 只写提交前像与 GPU 回读的差异及显式覆盖的并集，即使两者分成很多段。整页读改写会回写未归 GPU 的间隙，既可能丢失并行 CPU 更新，也可能因间隙不可写而失败。`title_backing::writeback_tests` 用夹在可写片段之间的只读宿主页区分两条路径。只保证不触及间隙，不裁决 CPU/GPU 对同一字节的冲突。

- 单次 `GuestCopy` 的源和目的重叠可经暂存保留源字节，但暂存不能证明**目的自身**的映射是单射。`copy_plan::GuestCopy::bind` 用提交固定页和页内半开区间检查；不同目的 VA 在同一物理页写到不同字节是合法的，若争写同一字节则在编码时拒绝整批。不同命令的有序覆盖不是这个反例。见 `copy_plan::alias_tests` 和 `page_queue::paged_buffer_copy_rejects_self_aliased_destination_before_submitting_prefix`。

## 已有的受限字节转换

- HLE `modules/xg.rs::layout` 输出的是 `victory_formats::texture::Texture::new` 建模的 data/depth/stencil plane；该模型没有 DCC 等元数据平面。GPU `resource::texture_layout` 使用同一构造器。因此默认 RTV 的 XG 单 plane、offset 0、与 GPU 计算大小相符，不是主平面无需元数据解释的独立证据。

- `victory_formats::texture::copy_uniform_rx_rgba8_blocks` 只处理独立证明为无压缩主平面的 2D `SW_64KB_R_X`、单层单 mip、单采样、4 字节 texel。按 GFX10 addrlib 的块索引，128×128 个 texel 占一块，块内 XOR 不改变块号。逐块验证全部四字节字相同（含填充）即可恢复紧密行；不同块不必同色。`uniform_rx` 集成测试有六个异色块及末块单字节反例。这不是通用逐像素寻址，未接入生产；510 页的 CPU 零观测不能替代平面描述、提交时的权威快照或同步证明。

- `copy_gfx10_1pipe_rx_rgba8` 可恢复非均匀块，但只实现公开 GFX10 非 RB+、单 pipe、256 B interleave、`pipeBankXor=0` 的上述主平面。在 `gfx10_rx` 夹具中，`Addr2ComputeSurfaceAddrFromCoord` 对同一幅 259×131 合成图的 texel (63,64) 在 1 和 2 pipe 下分别给出平面内字节偏移 `0x6a8c` 与 `0x6b8c`。布局码、行距、格式和大小不能选出方程。该函数不改变 `Layout::addressable()`，不用于判断游戏配置或旧色权威。

- `gfx10_rx::both_addrlib_metadata_modes_match_the_same_rgba8_main_plane` 独立引用同一 PAL 版本的 `Addr2ComputeSurfaceInfo` 正反对照：`noMetadata=1` 和 `requireMetadata=1` 的 3840×2160 R_X 主平面均为 33,423,360 B（510 页），对齐 64 KiB，mip 偏移 0；后一种能独立计算 163,840 B DCC 空间。该对照不识别实机的压缩模式，不能按这个大小搜寻或拼接游戏元数据。`64dd795` 的 510 页来自 VNME 按已建模平面生成的候选范围；节历史只在该范围内匹配，不测量完整分配。地址式创建保存基础 `ResourceDesc`，但实跑 RTV 摘要没有其原始 flags，合成的 DEFAULT 堆参数不是来宾压缩策略。当前没有可传递的独立平面契约；下一步若走字节路径，须先有能定位到**同一分配**的公开、独立且语义明确的格式/元数据契约，再证明提交时旧色版本；否则只追有证语义的原生生产者。

## 最小验证

- 使用仓库锁定的 Rust、Windows x64 MSVC、`--locked --features paged-title,diagnostics,blake3/pure`。新增诊断的定向测试过滤 `page_resource::native_history_tests`、`page_queue::copy_prefix_`；内容顺序的实像素测试为 `page_draw::paged_title_partial_color_clear_observes_submission_order`，仅相关改动时重跑。
- 构建按提交分目录，先核对程序的 SHA-256 和来源再实跑。不同程序或触发条件用独立日志；仅提炼脱敏证据，不保存游戏数据、原始地址或私人路径到仓库。尚未达到可玩验证，不以离线测试或菜单停留替代。

## 轮次 12 的生产判别 — 有序旧色生产者端到端反例

- 基线 `8294e04768d43ac29c9df2ee6298b83a605735e1`；整合提交 `8aa9317a2cb9c92a9cb726fea808a8a5ffb26427`，已普通推送并核对 `origin/refactor/unified-memory-engine` 同 SHA。改动限于 `page_draw.rs` 和 `page_queue.rs`，staged asset policy 通过。
- 新增无游戏数据的 D3D12/WARP 队列夹具：合成地址式 RGBA8 颜色目标由前置整面原生清除生产，再以真实 `SRC_ALPHA/INV_SRC_ALPHA` 混合绘制消费；Fence 完成后读回同时含底图红色和源绿色，证明旧色确实进入混合。无生产者、仅局部生产者、生产者晚于读者三种对照均在编码阶段拒绝，未放宽 `MissingTextureContent` 或 `TargetNotInitialized` 门槛。
- `ordered_color_disjoint` 现在拒绝空/倒置候选范围，新增固定多页目的、首尾页物理别名、普通操作共页、未捕获页的正反例。它只加强最终物理身份检查，不把候选 VA 或诊断历史当作内容权威。
- 定向验证使用锁定 Windows x64 MSVC 和 `--locked --offline`：有序颜色 2/2、混合旧色端到端 1/1、局部清除顺序 1/1、线性读取 2/2、UAV 预填充 2/2，共 8/8；测试程序 SHA-256 `0b046289ec1f06fa2c0bd05bf0bb5f964cd31b6ba990c222b72c5fe3f58e45e0`。
- 判别结论：正例证明现有提交内“完整原生生产者 → 同目标旧色读者”路径可运行，负例证明不能用局部覆盖、后置写或仅有不相交线索替代生产者。最近实机首错没有改变；`64dd795` 仍是 draw 15 `MissingTextureContent`，本轮未重复启动已知坏包，也没有把合成夹具说成实机修复。当前工程决策保持拒绝，继续追同一次提交中目标绑定、真实读者及可证明底图生产者；格式 29 的 `SW_64KB_R_X` 主平面/元数据和 510 页最新字节仍未证明。
- 下一步：从来宾调用录制还原 draw 15 的真实绑定和同提交前缀，优先找能在生产路径声明完整目的范围、物理身份和顺序的实际生产操作；若仍没有，审查最小缺失证据并换到数据平面/元数据来源，不再扩大历史日志。可控场景、10 分钟稳定性、峰值/帧时间和发行仍未验。

## 轮次 13 — 已知来宾前缀的目的描述符对照

- `64dd795` 的失败批次完整已编码前缀为索引 0–14；五个已识别原生写目的不匹配失败目标，唯一匹配目标的第 14 项增强纹理屏障（桌面候选形状）为 layout 6→2、access 128→8、flags 0。它的诊断降低和普通降低都只编码全局 UAV 屏障，不发布内容。归档直到首错没有通用 `command list` 缺槽告警；这不覆盖显式忽略的 `0x470`，其 ABI 与副作用仍未证。与桌面接口的槽号或调用位置相近都不能补足证明。
- `5e59087` 扩充 `page_draw::paged_title_blended_draw_requires_preceding_full_native_color_producer`：以合成地址式 sRGB、默认 R_X、蓝色优化清除值和四顶点条带，走真实来宾 RTV 描述符及清除处理器，再走增强屏障处理器和 VNME 队列。保持屏障和混合读者不变，只把红色整面清除的目的换成同尺寸的另一个默认私有堆 UNORM 对象，首错即为 draw 的 `MissingTextureContent`；清除实际目标时，Fence 后可读回红/绿混合、蓝为零及 0.75 线性 Alpha。开关 inventory 两条降低路径均通过。不把合成屏障编码当成来宾 ABI 的独立证明。
- 实机首错未重新采集、没有生产准入修复；可以排除**已知重放中**异对象清除、同对象布局转换或优化清除值被当作完整旧色。没有发现可修复的同批实体写入，不再扩大这条前缀的诊断。下一路线是取得实际目标独立的主平面/元数据契约（或能够产生原生版本的具体公开操作语义），再决定可否验证字节转换；若仍缺契约，保留拒绝，不因 CPU 零页放行。

## draw 15 的旧色依赖：混合与覆盖须分别证明

- `64dd795` 归档把实际宿主混合状态、单采样四顶点条带和覆盖目标的视口/裁剪绑定到失败 draw；不把附近的首次管线绑定和 shader 翻译记录绑定到该操作。即使知道实际 PS 的 `SV_Target0.a` 在签名中存在，也还没有各执行路径的输出值或覆盖证明。未确定 shader 翻译错误，不能根据时间邻近选 shader。
- `page_draw::paged_title_blended_draw_requires_preceding_full_native_color_producer` 现在用同一混合状态、几何、视口/裁剪，分别在红/蓝两种经真实清除产生的底图上读回全部 16 个像素：Alpha 0.5 的整面输出受底图影响；Alpha 1 的整面输出独立于底图；同为 Alpha 1 但在 PS 丢弃右半时，右半仍依赖底图。该 WARP 对照不是来宾 shader 翻译测试，不提供实机数据，不改变门禁。`ClosedDraw::encode` 本就不把无混合读依赖等同于可凭空定义完整目标：普通无目标混合依赖的 draw 也只有在明确弃旧时才能跳过旧内容，且不能因此发布完整版本。