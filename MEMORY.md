# 项目记忆

目标和验收边界见 [README](README.md)，工作方式见[开发原则](docs/development.md)。此页只留可复用、已核对的实现知识，不记轮次和原始实机资料。

- **启动与隔离。** `apps/victory/src/main.rs` 分流双击、`play` 和 `run`。`play.rs` 会先读 `%LOCALAPPDATA%/victory/victory-settings.txt`，再清理大部分继承的 `VICTORY_*` 变量。实机对照用独立 `LOCALAPPDATA`、`VICTORY_CONFIG`、`VICTORY_DATA_DIR` 和 `VICTORY_CACHE_DIR`；为 `play` 加临时诊断参数用 `--set`。这样不会复用用户存档；驱动缓存仍可能是热的。`run` 与 `play` 的调优默认值不同，不混作同一基线。
- **实机最小验证。** 启动日志应确认 `Xbox Series S`。开场、语言与首次设置可响应输入，只证明菜单正常；可玩要看同一 3D 场景里的实际画面和输入响应。按需保存同次送显路径的画面，统计资源时避免把频繁读回当作正常运行成本。
- **内存和搬运。** `crates/victory-gpu/src/host.rs` 的 DXGI local/non-local 使用量和预算是该进程在适配器的快照；`copystat.rs` 的私有提交与工作集是不同指标，不与 non-local 相加。纹理上传、回读计数覆盖各自路径，可能重叠，不相加、不冒充 PCIe/DRAM 带宽。先用同场景同配置找热点，再决定改动。
- **按窗口减小分配并不一定省内存。** XSS 热缓存对照都进入 500+ draw 的 3D 负载，但未验证可控。末次采样中，4 MiB 网格有 700 个 loose 镜像，合 3,004 MiB；仅将有长度请求的网格改为 1 MiB 后，镜像反增至 2,044 个、3,225 MiB，`E_OUTOFMEMORY` 更早出现，不采用。两次不是同帧采样；局部显存未满，系统提交和 DXGI non-local 已接近上限。更宽的注册缓冲查询不曾命中旧 64 项之外的候选，试验未合入。
- **独显支持按需后备，尚未接入游戏。** [硬件夹具](crates/victory-gpu/tests/sparse_backing.rs) 在 RTX 3060 Laptop、D3D12 Tiled Resources Tier 4 上保留 64 MiB GPU 缓冲地址，只用 128 KiB 显式堆映射两个 64 KiB tile，经真实队列 Fence 后读回已知字节。它不证明游戏 RAM/带宽收益、别名语义或整机可玩。[Xenia `95a5c3e`](https://github.com/xenia-project/xenia/commit/95a5c3e) 只参考按物理范围稀疏绑定、按需搬运及分别跟踪 GPU 写和 CPU 复制。首个接入点优先于整套迁移：挑实际占用明显、能证明内容来源的一类读取；`descriptor/host.rs` 的 `make` 与 `buffer_copy` 必须引用同一后备，不能只改刷新端。
- **分配失败不能留下持久占用。** `memory.rs` 的按地址目录或 arena 在 GPU 后备、fault 缓冲或编号失败前持有 `upload::Reservation`；失败释放尚未发布的 CPU 页。`shadow.rs` 只在完成后向全局扫描发布副本，显存额度随实际副本归还。不释放已录制或在途的后备，也不合并 CPU 与 GPU 各自的最后版本。`upload.rs` 的 Windows `VirtualQuery` 用例核实了失败边释放、成功边保留；未证明它是开场占用的主因。
- **启动期已释放的副本。** `apps/victory/src/run.rs` 在映射主映像后即释放原始文件和补丁缓冲，不把这两个已解决的副本当作新优化点。
