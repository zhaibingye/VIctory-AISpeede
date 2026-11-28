# 项目记忆

目标和验收边界见 [README](README.md)，工作方式见[开发原则](docs/development.md)。此页只留可复用、已核对的实现知识，不记轮次和原始实机资料。

- **启动与隔离。** `apps/victory/src/main.rs` 分流双击、`play` 和 `run`。`play.rs` 会先读 `%LOCALAPPDATA%/victory/victory-settings.txt`，再清理大部分继承的 `VICTORY_*` 变量。实机对照用独立 `LOCALAPPDATA`、`VICTORY_CONFIG`、`VICTORY_DATA_DIR` 和 `VICTORY_CACHE_DIR`；为 `play` 加临时诊断参数用 `--set`。这样不会复用用户存档；驱动缓存仍可能是热的。`run` 与 `play` 的调优默认值不同，不混作同一基线。
- **实机最小验证。** 启动日志应确认 `Xbox Series S`。开场、语言与首次设置可响应输入，只证明菜单正常；可玩要看同一 3D 场景里的实际画面和输入响应。按需保存同次送显路径的画面，统计资源时避免把频繁读回当作正常运行成本。
- **内存和搬运。** `crates/victory-gpu/src/host.rs` 的 DXGI local/non-local 使用量和预算是该进程在适配器的快照；`copystat.rs` 的私有提交与工作集是不同指标，不与 non-local 相加。纹理上传、回读计数覆盖各自路径，可能重叠，不相加、不冒充 PCIe/DRAM 带宽。先用同场景同配置找热点，再决定改动。
- **启动期已释放的副本。** `apps/victory/src/run.rs` 在映射主映像后即释放原始文件和补丁缓冲，不把这两个已解决的副本当作新优化点。
