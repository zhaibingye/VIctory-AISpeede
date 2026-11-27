# VNME 项目记忆

## 入口与约束

- [`docs/vnme.md`](docs/vnme.md) 是现行架构与验收入口；[`docs/unified-memory-engine.md`](docs/unified-memory-engine.md) 仅供回查历史。来宾命令由 `crates/victory-gpu/src/list.rs` 录制，在 `page_queue.rs` 编码、提交和退休；`page_draw.rs` 做附件读写准入，`page_resource.rs` 按宿主对象和子资源记内容权威。`resource.rs` 管宿主对象共享与换代，`title_backing.rs` 和 `victory-memory/src/physical.rs` 管捕获的物理后备。
- 同一对象的权威按本批 `pending` → 同队列在途窗口的 `SubmittedTail` → 已完成版本查询。单 mip 单 slice 的颜色纹理用整资源键，多子资源用精确键；不跨对象或跨 mip 继承。GPU Fence 真正观察完成且必要的字节合并成功后才发布完成版本。
- 以失败证据而不是纹理尺寸做判断：覆盖整个目标的视口和裁剪不证明 draw 实际写全；转换屏障本身不生产像素。弱前代、相同 VA、当前查找代和同物理页都是不同层面的线索，单独都不证明分配代、布局或最新像素归属。

## 当前实机阻塞

- `13eee34` 在 6 GB 显存 Windows 机上触发 `draw` 索引 15 的 `MissingTextureContent`：slot 0 / subresource 0，格式 29，3840×2160×1，1 mip，`reads_old_color=true`。新证据是未记录同键前代、没有已识别的兼容字节来源；失败前缀无同对象或虚拟范围重叠的原生写入。索引 14 增强纹理屏障未声明弃旧。这不排除更早的提交、不同 VA 的物理别名或字节来源。
- `b40a245` 手动按 A 复现同一首错。`[VNME_PREFIX_COPY_INVENTORY]` 中来宾字节复制、显式纹理上传和纹理复制均为零；只涵盖**失败且未提交**批次的已编码前缀。`[VNME_NATIVE_WRITE_HISTORY]` 在拒绝点识别到目标 510 个物理页；保留的 128 个 Fence 内，256 个原生写入候选均归类为默认私有堆，未发现同对象、VA 重叠或共页；另有 600 个 Fence 的候选已淘汰。两次定时模拟 A 未触发首错，不能算通过；复现以可见继续提示时的手动输入为准。
- 后续诊断按相同对象候选和提交前的物理快照压缩重复原生记录，统计摘要组与 Fence 数；不在失败时重新解释历史 VA。目标只捕获一次，区分 XMem section 与跟踪的私有 VM 页，用同一快照比对原生和 `[VNME_BYTE_WRITE_HISTORY]` 字节写回候选。字节历史在 Fence 完成且读回合并成功后才登记，仍不涵盖 CPU、DStorage、在途或淘汰的写入。整资源候选范围和摘要变化不能证明有效 texel 覆盖。

## 最小验证

- 使用仓库锁定的 Rust、Windows x64 MSVC、`--locked --features paged-title,diagnostics,blake3/pure`。新增诊断的定向测试过滤 `page_resource::native_history_tests`、`page_queue::copy_prefix_`；内容顺序的实像素测试为 `page_draw::paged_title_partial_color_clear_observes_submission_order`，仅相关改动时重跑。
- 构建按提交分目录，先核对程序的 SHA-256 和来源再实跑。不同程序或触发条件用独立日志；仅提炼脱敏证据，不保存游戏数据、原始地址或私人路径到仓库。尚未达到可玩验证，不以离线测试或菜单停留替代。
