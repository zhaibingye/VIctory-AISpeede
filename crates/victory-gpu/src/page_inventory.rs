//! 汇总已经录制的命令，不继续翻译或执行被拒绝的提交。

use std::sync::OnceLock;
use crate::{host::HostList, page_shader::UnloweredShader};

// 每个列表及整批分别限额；重复命令只累计次数。
const LIMIT: usize = 96;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Observation {
    Command(&'static str),
    Pipeline {
        command: &'static str,
        shaders: String,
        colors: Vec<i32>,
        depth: Option<i32>,
        rejection: Option<UnloweredShader>,
    },
    Rejection(String),
}

impl Observation {
    fn bucket(&self) -> usize {
        // 为所有三类事实各留空间，成功的管线不能占满拒绝原因的配额。
        match self {
            Self::Rejection(_) | Self::Pipeline { rejection: Some(_), .. } => 0,
            Self::Pipeline { .. } => 1,
            Self::Command(_) => 2,
        }
    }
}

#[derive(Default)]
pub(crate) struct Inventory {
    entries: [(Vec<(Observation, u64)>, u64); 3],
}

impl Inventory {
    pub(crate) fn clear(&mut self) { *self = Self::default(); }

    pub(crate) fn observe(&mut self, entry: Observation) {
        self.add(&entry, 1);
    }

    fn add(&mut self, entry: &Observation, count: u64) {
        let (entries, dropped) = &mut self.entries[entry.bucket()];
        if let Some((_, n)) = entries.iter_mut().find(|(existing, _)| existing == entry) {
            *n = n.saturating_add(count);
        } else if entries.len() < LIMIT {
            entries.push((entry.clone(), count));
        } else {
            *dropped = dropped.saturating_add(count);
        }
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        for (i, (entries, dropped)) in other.entries.iter().enumerate() {
            for (entry, count) in entries {
                self.add(entry, *count);
            }
            let skipped = &mut self.entries[i].1;
            *skipped = skipped.saturating_add(*dropped);
        }
    }
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::switch::on("VICTORY_VNME_INVENTORY"))
}

pub(crate) fn report(lists: &[&HostList]) {
    if !enabled() {
        return;
    }
    let mut result = Inventory::default();
    for list in lists {
        list.summarize_inventory(&mut result);
    }
    let dropped = result.entries.iter().map(|(_, dropped)| *dropped)
        .fold(0u64, u64::saturating_add);
    tracing::warn!(lists = lists.len(), dropped, "[VNME_BATCH_INVENTORY] 被拒绝的提交：已录制命令盘点");
    for (entries, _) in result.entries {
        for (entry, count) in entries {
            tracing::warn!(?entry, count, "[VNME_BATCH_ITEM] 盘点条目");
        }
    }
}

#[test]
fn merges_repeats_without_crowding_out_rejections() {
    let mut first = Inventory::default();
    let mut later = Inventory::default();
    let command = Observation::Command("Dispatch");
    first.observe(command.clone());
    later.observe(command);
    for n in 0..=LIMIT {
        later.observe(Observation::Pipeline {
            command: "Dispatch",
            shaders: n.to_string(),
            colors: Vec::new(),
            depth: None,
            rejection: None,
        });
    }
    later.observe(Observation::Rejection("NotClosed".into()));
    first.merge(&later);
    assert_eq!(first.entries[0].0.len(), 1);
    assert_eq!(first.entries[1].0.len(), LIMIT);
    assert_eq!(first.entries[1].1, 1);
    assert_eq!(first.entries[2].0[0].1, 2);
}
