//! `victory`: the command-line front end.
//!
//! Every command reads the user's own files and writes only to stdout or to
//! the user's cache directory (outside the repository).

mod census;
mod inspect;
mod prebuild;
mod run;
mod shader;
mod translate;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "victory",
    version,
    about = "在 Windows 上原生运行 Xbox Series 游戏"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 查看 PE 文件的文件头、节、导入表、TLS 和构建标识。
    Inspect {
        file: PathBuf,
        /// 列出所有导入函数。
        #[arg(long)]
        imports: bool,
        /// 列出所有导出函数。
        #[arg(long)]
        exports: bool,
    },
    /// 汇总指定路径下 PE 文件的导入项并检查支持情况；未指定路径时使用配置中的游戏和测试目录。
    Census {
        paths: Vec<PathBuf>,
        /// 宿主直通不计入已支持项。
        #[arg(long)]
        strict: bool,
        /// 列出缺失的 `dll!function`，按使用次数降序排列。
        #[arg(long)]
        missing: bool,
    },
    /// 列出导入库中的 `dll!export` 对应关系。
    LibExports {
        lib: PathBuf,
        /// 逐项显示 `dll!export`。
        #[arg(long)]
        list: bool,
    },
    /// 查找文件、目录或未压缩引擎归档里的着色器容器，并显示它们的结构。
    ShaderInfo {
        paths: Vec<PathBuf>,
        /// 逐项显示着色器容器及包内管线。
        #[arg(long)]
        verbose: bool,
        /// 同时显示解码后的原生指令。
        #[arg(long)]
        disasm: bool,
        /// 改为列出着色器数据库的技术、通道、着色器标识与通道状态。
        #[arg(long)]
        effects: bool,
    },
    /// 翻译找到的着色器容器和 SGD5 文件，并报告尚不支持的情况。
    Translate {
        paths: Vec<PathBuf>,
        /// 用 dxil.dll 验证生成的 DXIL，优先使用 `VICTORY_DXIL` 指定的版本。
        #[arg(long)]
        validate: bool,
        /// 建立宿主 D3D12 管线；与 `--cache` 同用时，预编译着色器包中的管线，只补齐当前 GPU 与驱动缺少的项。
        #[arg(long)]
        pipelines: bool,
        /// 逐条显示程序及翻译结果。
        #[arg(long)]
        list: bool,
        /// 工作线程数，默认使用全部逻辑核心。
        #[arg(long)]
        jobs: Option<usize>,
        /// 预编译着色器并写入 `run` 使用的缓存；加上 `--pipelines` 也会预编译管线。
        #[arg(long, conflicts_with_all = ["validate", "list"])]
        cache: bool,
    },
    /// 准备、映射并绑定游戏 EXE；使用 `--execute` 运行。
    Run {
        exe: PathBuf,
        /// 调用未实现的导入项时的处理方式。
        #[arg(long, value_enum, default_value_t = Unimplemented::Trap)]
        unimplemented: Unimplemented,
        /// 禁用宿主直通，要求所有导入项均由运行时实现。
        #[arg(long)]
        strict: bool,
        /// 实际启动游戏；未指定时，在完成绑定后停止。
        #[arg(long)]
        execute: bool,
        /// 兼容参数，目前仅支持 Series S。
        #[arg(long, value_enum, hide = true)]
        console: Option<Console>,
        /// `--` 后的参数传给游戏。游戏可以读取完整命令行。
        #[arg(last = true)]
        title_args: Vec<String>,
    },
    /// 显示并检查生效的配置。
    Config,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unimplemented {
    Trap,
    Zero,
}

#[derive(Clone, Copy, ValueEnum)]
enum Console {
    SeriesS,
}

fn main() -> Result<()> {
    victory_core::logging::init("info");
    // The audio recorder `run` starts for `VICTORY_AUDIO_CAPTURE`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if victory_hle::modules::audio::capture::child() {
        return Ok(());
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect {
            file,
            imports,
            exports,
        } => inspect::run(&file, imports, exports),
        Command::Census {
            paths,
            strict,
            missing,
        } => census::run(paths, strict, missing),
        Command::LibExports { lib, list } => inspect::lib_exports(&lib, list),
        Command::ShaderInfo {
            paths,
            verbose,
            disasm,
            effects,
        } => shader::run(paths, verbose || disasm, disasm, effects),
        Command::Translate {
            paths,
            jobs,
            pipelines,
            cache: true,
            ..
        } => translate::warm(paths, jobs, pipelines),
        Command::Translate {
            paths,
            validate,
            pipelines,
            list,
            jobs,
            cache: false,
        } => translate::run(paths, validate, pipelines, list, jobs),
        Command::Run {
            exe,
            unimplemented,
            strict,
            execute,
            console: _,
            title_args: _,
        } => {
            let code = run::run(&exe, unimplemented, strict, execute)?;
            if code != 0 {
                // Preserve all 32 bits of the Windows process status.
                std::process::exit(code as i32);
            }
            Ok(())
        }
        Command::Config => show_config(),
    }
}

fn show_config() -> Result<()> {
    let cfg = victory_core::Config::load()?;
    let status = |p: &Option<PathBuf>| match p {
        Some(p) if p.exists() => format!("{}", p.display()),
        Some(p) => format!("{}  （不存在）", p.display()),
        None => "（未设置）".to_owned(),
    };
    println!(
        "配置文件： {}",
        cfg.source
            .as_ref()
            .map_or("（无）".into(), |p| p.display().to_string())
    );
    println!("game_dir    : {}", status(&cfg.game_dir));
    println!("cache_dir   : {}", cfg.cache_dir.display());
    println!("data_dir    : {}", cfg.data_dir.display());
    println!("fidelityfx  : {}", status(&cfg.fidelityfx_dir));
    for t in &cfg.test_titles {
        println!("test_title  : {}", status(&Some(t.clone())));
    }
    for p in cfg
        .game_dir
        .iter()
        .chain(&cfg.test_titles)
        .chain([&cfg.cache_dir, &cfg.data_dir])
    {
        if let Err(e) = victory_core::ensure_outside_repo(p) {
            println!("路径违反资料存放规则： {e}");
        }
    }
    Ok(())
}

/// Paths given on the command line, or the configured game + test titles.
pub(crate) fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !paths.is_empty() {
        return Ok(paths);
    }
    let cfg = victory_core::Config::load()?;
    let all: Vec<PathBuf> = cfg.game_dir.into_iter().chain(cfg.test_titles).collect();
    anyhow::ensure!(
        !all.is_empty(),
        "no paths given and none configured (see victory.example.toml)"
    );
    Ok(all)
}

#[cfg(test)]
mod console_args_tests {
    use super::{Cli, Parser};

    #[test]
    fn series_s_is_the_only_console() {
        let parse = |flags: &[&str]| {
            Cli::try_parse_from(["victory", "run", "example.exe"].into_iter().chain(flags.iter().copied()))
        };
        assert!(parse(&["--execute"]).is_ok());
        assert!(parse(&["--console", "series-s"]).is_ok());
        assert!(parse(&["--console", "series-x"]).is_err());
    }
}
