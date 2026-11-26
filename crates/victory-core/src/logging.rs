//! VIctory 日志：终端输出与可选 UTF-8 文件输出。
//! 使用 VICTORY_LOG（或 RUST_LOG）筛选；调试和跟踪需要 diagnostics。
//! 设置 VICTORY_LOG_FILE 可绕过终端或管道的解码直接保存日志。

use std::{fs::OpenOptions, path::PathBuf, sync::Mutex};

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// 当前构建允许的最详细日志级别。
const CAP: LevelFilter = if cfg!(feature = "diagnostics") {
    LevelFilter::TRACE
} else {
    LevelFilter::INFO
};

pub fn init(default_level: &str) {
    let filter = std::env::var("VICTORY_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .ok()
        .and_then(|v| EnvFilter::try_new(v).ok())
        .unwrap_or_else(|| EnvFilter::new(default_level));
    let file = std::env::var_os("VICTORY_LOG_FILE")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .and_then(|path| {
            let result = crate::ensure_outside_repo(&path)
                .map_err(anyhow::Error::from)
                .and_then(|()| OpenOptions::new().create(true).append(true).open(&path)
                    .map_err(anyhow::Error::from));
            match result {
                Ok(file) => Some(Mutex::new(file)),
                Err(error) => {
                    eprintln!("无法打开 UTF-8 日志 {}：{error}", path.display());
                    None
                }
            }
        });
    let _ = tracing_subscriber::registry()
        .with(CAP)
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(std::io::stderr),
        )
        .with(file.map(|file| {
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_ansi(false)
                .with_writer(file)
        }))
        .try_init();
}
