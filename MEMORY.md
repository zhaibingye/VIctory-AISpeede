# VNME 项目记忆

这里记录跨任务复用的实现约束、已验证的反例和测试入口，不按轮次追加流水账。[开发原则与证据止损](docs/development.md)规定如何选路；[现行架构与验收](docs/vnme.md)说明系统职责和交付门槛；[实机证据](docs/vnme-evidence.md)记录实际构建、触发条件和观察边界。

## 定位表

| 遇到的问题 | 从哪里查 |
| --- | --- |
| 来宾命令在哪里录制、提交和退休？ | [`list.rs`](crates/victory-gpu/src/list.rs) → [`page_queue.rs`](crates/victory-gpu/src/page_queue.rs)，来宾 Fence 等待入口见 [`title_timeline.rs`](crates/victory-gpu/src/title_timeline.rs)。 |
| RTV 是否要读旧色？谁能发布新版本？ | [`page_draw.rs`](crates/victory-gpu/src/page_draw.rs) 决定附件读写准入；[`page_resource.rs`](crates/victory-gpu/src/page_resource.rs) 记对象/子资源权威；[`resource.rs`](crates/victory-gpu/src/resource.rs) 管宿主对象共享与换代。 |
| 当前 VA 映到谁？捕获的后备是谁、谁能写？ | [`xmem.rs`](crates/victory-hle/src/modules/xmem.rs) 管 section、映射和准入；[`physical.rs`](crates/victory-memory/src/physical.rs) 管物理页身份和访问声明；[`title_backing.rs`](crates/victory-gpu/src/title_backing.rs) 捕获后备、暂存并合并写回；[`dstorage.rs`](crates/victory-hle/src/modules/dstorage.rs) 有 DirectStorage 的写窗口。 |
| 共享纹理读取或复制为什么在入队前拒绝？ | [`page_texture.rs`](crates/victory-gpu/src/page_texture.rs) 和 [`page_uav.rs`](crates/victory-gpu/src/page_uav.rs) 验证读取范围，[`copy_plan.rs`](crates/victory-gpu/src/copy_plan.rs) 绑定复制区间；[`page_queue.rs`](crates/victory-gpu/src/page_queue.rs) 检查本批及先前提交的冲突。 |
| R_X 计算大小为何不等于读得出像素？ | [`xg.rs`](crates/victory-hle/src/modules/xg.rs) 的 `layout` 和 GPU 的 `resource::texture_layout` 同依赖 [`victory_formats::texture`](crates/victory-formats/src/texture.rs)；独立对照入口是 `victory-formats` 的 `gfx10_rx` 夹具。 |

## 长期不变量与反例

### 页身份、映射生命周期与 RAM 准入

- 当前 VA 查找、固定的 64 KiB 物理页身份和某一时刻的字节版本是三件事。提交按捕获的 section 页及租约工作，不按历史 VA 回查。真实 Windows 对照中，未被写过的新节首提交为零，但物理写可早于首次来宾绑定；在旧页已固定且同 VA 换节后，旧 `PageWindow` 仍可写旧节。随后对固定页的读取看到的是后写值，不是 pin 时的前像。弱前代、同 VA、共页或 CPU 当前读到零，均不能单独证明最新像素或分配布局。
- 原生 Windows `paged-title` 用 `SEC_RESERVE` 建 section。`DeferredRam` 按**同一 section 内净新增提交页**预扣额度；别名不二次计费，宿主提交失败仅退回此次新增额度。解绑一个 VA 不释放仍存活的 section；同 VA 换节不是原节复用。拒绝日志的 `requested_mib` 是区间长度，不是净新增量。释放 Loader 的临时文件/补丁缓冲可降低启动峰值，但不释放 XMem 租约或改变计费公式。
- `CopyPageMappingsBatchX` 只复制受管 XMem 视图关系；登记页池不导入已存在的私有分配。`map_alias` 要求已登记的 section 源和可替换的 Windows placeholder 目的。同节双 VA 可以看到彼此写入并捕获为同页；活跃私有堆即使对齐也不是合法源，普通已提交私有目的也不会被替换。这只排除当前实现下直接共用字节，不排除间接复制或其它语义。

### 队列顺序、权威与真实完成

- 旧内容按同一键依序查本批 `pending` → 同队列在途窗口的 `SubmittedTail` → 已完成版本。单 mip 单 slice 的颜色纹理可用整资源键，多子资源用精确键；尺寸、相邻命令和跨对象版本号不建立继承关系。相同布局或物理身份也不能代替内容权威。
- 已编码前缀不等于已入队，已入队不等于已完成。队列顺序可服务后续同队列消费者；页、描述符及在途资源所需的保活不能提前撤销，也不能用 CPU 计时、命令记录或诊断历史代替实际 Fence。只有观察到相关真实 GPU Fence 完成、必要的字节回读与合并成功之后，才能发布完成版本。不能以静默回退 Shadow Buffer 来掩盖准入失败。
- `PresentX` 在逻辑队列运行到该操作时，只对实际消费的平面处理已发布的 `CompatibleBytes`；在同一个 Direct Queue 锁内，经 `ResourcePlan::require_native` 转成同版本的原生表示、恢复资源状态，再录制并执行送显读取。没有已发布 CPU 版本时不为这一步上传；这不证明未追踪的指针写或未知别名有效。合成器建立窗口和调用 DXGI `Present` 不持这个锁。
- `title_backing::merge_page` 只写提交前像与 GPU 回读的**差异**与**显式覆盖**的并集，不按整页写回。中间间隙可能有并行 CPU 更新，也可能是只读宿主页；整页读改写会丢掉更新或失败。碎片写回只保证不触及这些间隙，不裁决 CPU/GPU 同字节冲突。

### 写窗口、快照和共享读取

- `CapturedUploads::stage` 在拷第一张页前通过 `physical::access::try_snapshot` 一次登记整批固定的 section 页，持有到最后一页拷完。`PageWindow`、`write_physical` 用相同身份登记完整写窗口：写者等已有快照，新快照遇到相交写窗口则在入队前报 `PhysicalWriterDuringSnapshot`。同址换节不冲突；逐页加锁会把跨页写的两侧拷成两个版本。协议不覆盖任意来宾 CPU 或原生 GPU 写，也不为纹理发布权威。
- DirectStorage 的受管 `paged-title` 裸 VA fill 在整个同步填充（含部分失败）期间持有标题视图共享门锁，`XMemMapPhysicalPages` 和 `map_alias` 在独占侧重映射；改一页也可能暂时撤下整个宿主视图。fill 在门锁内解析所有相交 section 页，释放 `PHYSICAL` 后才等待快照；上传暂存不取视图门锁。未知或私有字节不获得 section 身份。fill 可以并行，重映射可能等无关 I/O；写窗口不裁决暂存之后的同字节写回。
- `record_write_at_with` 把「可能写共享后备」与「可发布整面原生颜色版本」分开。弃旧后的半面清除确实写入，但整面仍为 `Discarded`。共享读取在原生入队前要查本批和此前提交的未分类原生写；拒绝是保守裁决，不证明相交，更不提供旧色。
- 共享体纹理 UAV 虽是输出，`ShaderTextureUav::prepare` 仍需预填充整面以保留着色器未写的 texel。它与共享 SRV 经 `SharedTextureBytes::gather_into` 验证各有效区间的独立后备并登记读取；`verify_shared_volume` 的视图/范围检查不能替代它。`TextureInput::Linear` 也可来自地址式 SRV；带外部别名时要逐行验证有效字节（不含 pitch 填充）并走同一读取门禁。不同对象、不同 VA 仍可指向同一固定页，只看读者自身的版本会漏掉写者。
- 单次 `GuestCopy` 的源/目的重叠可以暂存源解决，但目的映射仍须是单射。`GuestCopy::bind` 按固定页和页内半开区间检查：不同目的 VA 写同页的不同字节合法，争写同一字节则编码时拒绝整批；不同命令之间的有序覆盖不是这种冲突。

### 旧色需要生产者，完整覆盖需要单独证明

- 视口/裁剪覆盖整面不等于 draw 实际写全；`ADD(SRC_ALPHA, INV_SRC_ALPHA)` 不能只看几何或 PS 输出签名就忽略底图。合成对照中，Alpha 0.5 依赖旧色；Alpha 1 **且真正写全**才与底图无关；Alpha 1 但丢弃半面，未写区域仍保留底图。`ClosedDraw::encode` 不把「无混合读依赖」当成完整初始化：写颜色的普通 draw 仅在无目标读依赖且明确弃旧时跳过旧内容检查，并不发布完整版本；全零颜色写掩码则既不消费也不生产内容。
- 同一目标的地址式/RTV 创建、优化清除值和布局屏障不是像素生产者。合成 Windows 映射对照中，创建逐字节保留零或非零后备；后接布局屏障仍不能为读者提供原生旧色。有效的正例是**在读者之前对实际目标的整面原生清除**，再由真实混合命令消费，Fence 后读回验证。无写者、异对象同尺寸清除、局部清除、写者在读者之后均不能替代；`ordered_color_disjoint` 还需要完整捕获的物理目的，不能拿空范围或仅 VA 不交来证明安全。

### R_X 布局与独立证据

- XG `layout` 和 GPU `texture_layout` 都使用 `victory_formats::texture::Texture::new`。模型描述 data/depth/stencil，不含 DCC 等元数据平面。同报单 plane、零偏移和相同大小不是两份独立证据。公开 GFX10 addrlib 夹具中，要求元数据和明确无元数据可以有完全相同的 R_X 主平面几何；相同布局码、格式、行距与大小也不能区分不同 pipe 配置的块内方程。用模型算出的候选页范围不是整个分配的测量。
- `copy_uniform_rx_rgba8_blocks` 只处理已**独立证明无压缩**的 2D `SW_64KB_R_X` 主平面，单层、单 mip、单采样、4 字节 texel。块内含填充的每个四字节字都须相同，各块颜色可不同；这不是通用逐 texel 解码。`copy_gfx10_1pipe_rx_rgba8` 能恢复非均匀块，但限定公开 GFX10 非 RB+、单 pipe、256 B interleave、`pipeBankXor=0`。这些夹具不扩大生产 `Layout::addressable()`，也不能用 CPU 零页替代同一分配的布局/元数据契约和提交时的权威快照。
- XG CPU 复制和 D3D12 上传是不同路径。合成默认 R_X RTV 的有效 XG copy-in 返回 `E_NOTIMPL` 且不改字节；显式 row-major 对照可写入。这只排除该入口在该有效布局上静默成功，不排除其它写入。日志没告警也不能反推未调用；生产准入仍需要同一分配的独立内容契约和权威字节，或有确证语义、目的和顺序的完整生产者。

## 定向测试与复现边界

下表是改到相关机制时的测试名或过滤前缀，不是每次都跑的清单。Windows 原生实测夹具使用仓库锁定的 Rust、Windows x64 MSVC、`--locked --features paged-title,diagnostics,blake3/pure`；只选与假设有关的测试。

| 要区分的行为 | 测试入口 |
| --- | --- |
| 净新增、宿主拒绝回滚、同址重试；固定页非冻结、受管节别名 vs 私有堆 | `xmem::tests::deferred_native_`、`xmem::tests::native_reserved_section_first_commit_is_zero_but_pinned_remap_keeps_bytes`、`xmem::tests::managed_xmem_alias_shares_section_not_live_private_memory` |
| 写窗口整批保护、换节隔离、裸 VA 部分失败仍保护；入队前拒绝 | `xmem::snapshot_tests`、`raw_va_fill_excludes_snapshots_even_on_partial_error`、`page_queue::paged_copy_snapshot_rejects_physical_writer_before_enqueue` |
| 按片段写回，保留/避开间隙；单次复制的同页非重叠 vs 争写 | `title_backing::writeback_tests`、`copy_plan::alias_tests`、`page_queue::paged_buffer_copy_rejects_self_aliased_destination_before_submitting_prefix` |
| 弃旧局部清除仍挡共享读；线性 SRV 跨对象别名；体 UAV 未证明的预填充范围 | `page_queue::discarded_partial_color_clear_blocks_shared_texture_read`、`page_queue::linear_read_tests`、`page_uav::prefill_tests` |
| WARP 实像素读回：提交顺序、完整原生生产者、Alpha 与覆盖、地址式创建不生产旧色 | `page_draw::paged_title_partial_color_clear_observes_submission_order`、`page_draw::paged_title_blended_draw_requires_preceding_full_native_color_producer` |
| Windows 交互桌面：双层 `PresentX` 显式 CPU 写前后，及同源 `CopyResource` 真 Fence 读回 | `queue::present_fixture::present_x_resolves_explicit_texture_write_before_copy_resource`（定向 `--ignored`） |
| 有序颜色路径的目的范围须完整捕获；别名与空范围反例 | `page_queue::ordered_color_disjoint_requires_complete_physical_destination` |
| 均匀块末尾反例、非均匀块与 pipe 选择、同主平面不同元数据要求、XG CPU 复制边界 | `victory-formats` 的 `uniform_rx`、`gfx10_rx`（含 `both_addrlib_metadata_modes_match_the_same_rgba8_main_plane`）；`modules::xg::tests::copy_into_rejects_default_render_target_layout_without_writing` |
| 只改诊断时验证历史分类和失败前缀报告 | `page_resource::native_history_tests`、`page_queue::copy_prefix_` |

- 未提交前缀的复制/声明脏页只说明计划，不是写入；有限历史的无命中不是无写者。对 DirectStorage，确定物理身份要靠写侧保活的 `Slice`；裸 VA 前后快照可能采样不全，端点一致不证明中途没重绑。直接写成功只确认实际写入长度，原地失败仅是可能写过。
- 实机对照要核对实际运行的提交和可执行文件 SHA-256，分开记录构建与触发条件。确认可见继续提示后确实触发 A；定时输入未触发、只停留开始画面或离线夹具通过，都不算通过。送显结果只能证明观察到的场景。按[实机证据](docs/vnme-evidence.md)留脱敏结论并引用 Git 提交，不入库游戏资产、裸地址、敏感数据或个人机器路径。

## 当前状态指针

- VNME 最近实机基准 `8c7269a`：按 A 后 draw 15 首拒 `MissingTextureContent`，未证明实际目标有可读的完整旧色。
- 独立 `fxceless/victory-fork@2eade68`：同机、同游戏入口、Series S 到达语言选择（实证记录 `4cce33f`）。未确认同一 draw、有效底图或可控场景，不是可玩验证。后续对照按来宾描述、实际 RTV 绑定、混合与相对顺序配对，不按 draw 编号、尺寸或地址猜对应；证据和限制以[实机证据](docs/vnme-evidence.md)为准。
