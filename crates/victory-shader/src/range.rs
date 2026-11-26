//! Upper bounds of the index of register-relative moves (`v_movrel*`,
//! `v_swaprel_b32`), proven on the per-lane form, so that a move names only
//! the registers its M0 can reach rather than every one up to the program's
//! VGPR count.
//!
//! A value is bounded when it is a constant; a copy (`s_mov_b32`,
//! `v_mov_b32`, or an unguarded `v_readfirstlane_b32`: the first active
//! lane is then a lane that runs there) of a bounded value; `min`, `and` or
//! `med3` (unsigned) with a bounded operand; a sum of bounded values that
//! cannot wrap; or a phi whose every incoming value is bounded on its edge,
//! where an unsigned compare of that value with a bounded one (`v_cmp*`,
//! `s_cmp*`: `lt`, `le`, `eq` true or `gt`, `ge` false) that every lane
//! taking the edge has passed also bounds it: a loop counter against a
//! bounded limit. A VGPR write under a guard keeps the old value in the
//! other lanes, so it is bounded only if both are. Anything else (memory,
//! signed compares, what a relative move writes) is unbounded, and a move
//! with an unbounded index keeps today's range.

use std::collections::{HashMap, HashSet};

use crate::ShaderError;
use crate::gfx10::{Fields, Operand};
use crate::ir::{self, EXEC, Inst, M0, Program, Reg, Value};
use crate::structure::{self, Atom, Bool, LaneExit, Structured};

/// [`ir::build`] and [`structure::structure`], built again with the register
/// ranges of relative moves bounded where their index is.
///
/// A move can write the registers its own index comes from (the full range
/// of a `v_movreld` covers a loop's counter and limit), so the bounds are
/// found as an induction: assume each move reaches only its first register,
/// widen to what the analysis finds under the assumption until the
/// assumption implies itself. Then no run takes a move past it (the first
/// move that would has run, until then, as the assumption says), and what
/// the analysis found holds.
pub fn build(
    code: &[u8],
    wave64: bool,
    vgprs: u16,
) -> Result<(Program, Result<Structured, ShaderError>), ShaderError> {
    let p = ir::build(code, wave64, vgprs)?;
    let st = structure::structure(&p);
    let mut assumed: HashMap<usize, u8> = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| relative(i))
        .map(|i| (i.machine.offset, 0))
        .collect();
    if st.is_err() {
        assumed.clear();
    }
    // Each round grows a bound or drops a move; few programs need more than two.
    for _ in 0..8 {
        if assumed.is_empty() {
            break;
        }
        let q = ir::build_bounded(code, wave64, vgprs, &assumed)?;
        let Ok(s) = structure::structure(&q) else {
            break;
        };
        let found = relative_bounds(&q, &s);
        let proven: HashMap<usize, u8> = assumed
            .iter()
            .filter_map(|(&o, &m)| Some((o, u8::try_from(found[&o]).ok().filter(|&f| f <= m)?)))
            .collect();
        if proven.len() == assumed.len() {
            if proven == assumed {
                return Ok((q, Ok(s)));
            }
            let q = ir::build_bounded(code, wave64, vgprs, &proven)?;
            let s = structure::structure(&q);
            return Ok((q, s));
        }
        assumed = assumed
            .iter()
            .filter_map(|(&o, &m)| Some((o, u8::try_from(found[&o]).ok()?.max(m))))
            .collect();
    }
    Ok((p, st))
}

fn relative(inst: &Inst) -> bool {
    matches!(
        inst.name(),
        "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" | "v_movrelsd_2_b32" | "v_swaprel_b32"
    )
}

/// The largest M0 each relative move can see, by instruction offset
/// (`u32::MAX`: unbounded).
fn relative_bounds(p: &Program, st: &Structured) -> HashMap<usize, u32> {
    let mut r = Ranges::new(p, st);
    // Copies of a block (irreducible loops) share offsets: the largest wins.
    let mut out: HashMap<usize, u32> = HashMap::new();
    for inst in p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| relative(i))
    {
        let hi = inst.read(Reg::S(M0)).map_or(u32::MAX, |m0| r.hi(m0));
        let e = out.entry(inst.machine.offset).or_insert(0);
        *e = (*e).max(hi);
    }
    out
}

#[derive(Clone, Copy)]
enum Def {
    Phi(usize, usize),
    Inst(usize, usize),
}

pub(crate) struct Ranges<'a> {
    p: &'a Program,
    st: &'a Structured,
    defs: HashMap<Value, Def>,
    memo: HashMap<Value, u32>,
    /// Values being bounded: met again, a cycle, unbounded.
    busy: HashSet<Value>,
}

impl<'a> Ranges<'a> {
    pub(crate) fn new(p: &'a Program, st: &'a Structured) -> Self {
        let mut r = Self {
            p,
            st,
            defs: HashMap::new(),
            memo: HashMap::new(),
            busy: HashSet::new(),
        };
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                r.defs.insert(phi.value, Def::Phi(b, i));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for w in &inst.writes {
                    r.defs.insert(w.1, Def::Inst(b, i));
                }
            }
        }
        r
    }

    /// 只给硬件传入的 SSA 值设界。对同一寄存器的后续写入仍按指令证明。
    pub(crate) fn bound_input(&mut self, reg: Reg, upper: u32) {
        if let Some(&(_, value)) = self.p.inputs.iter().find(|&&(input, _)| input == reg) {
            self.memo.insert(value, upper);
        }
    }

    /// A conservative unsigned byte bound and independent dword alignment
    /// proof. Only plain moves/phis/masks are used to infer low bits.
    pub(crate) fn dword_offset_hi(&mut self, value: Value) -> Option<u32> {
        if !self.dword_aligned(value, &mut HashSet::new(), 64) {
            return None;
        }
        let hi = self.hi(value);
        (hi != u32::MAX).then_some(hi & !3)
    }

    fn dword_aligned(&self, v: Value, active: &mut HashSet<Value>, depth: usize) -> bool {
        if depth == 0 || !active.insert(v) {
            return false;
        }
        let result = match self.defs.get(&v).copied() {
            Some(Def::Phi(b, i)) => {
                let args = &self.p.blocks[b].phis[i].args;
                !args.is_empty() && args.iter().all(|&a| self.dword_aligned(a, active, depth - 1))
            }
            Some(Def::Inst(b, i)) => {
                let inst = &self.p.blocks[b].insts[i];
                let reg = inst.writes.iter().find_map(|&(r, x)| (x == v).then_some(r));
                let next = reg.is_some() && reg == dst(inst)
                    && self.aligned_result(inst, active, depth - 1);
                next && !(matches!(reg, Some(Reg::V(_))) && self.st.guards[b][i].is_some()
                    && !reg.and_then(|r| inst.read(r)).is_some_and(|old| {
                        self.dword_aligned(old, active, depth - 1)
                    }))
            }
            None => false,
        };
        active.remove(&v);
        result
    }

    fn aligned_result(&self, inst: &Inst, active: &mut HashSet<Value>, depth: usize) -> bool {
        let operands = match &inst.fields {
            Fields::Valu(v) if v.sdwa.is_none() && v.dpp.is_none()
                && !v.clamp && v.omod == 0 && v.neg | v.abs | v.opsel == 0 => &v.src,
            Fields::Sop { src, .. } => src,
            _ => return false,
        };
        let constant = |op: &Operand| match *op {
            Operand::Const(c @ 128..=192) => Some(u32::from(c - 128)),
            Operand::Literal(x) => Some(x),
            _ => None,
        };
        match (inst.name(), operands.as_slice()) {
            ("v_and_b32" | "s_and_b32", [a, b]) => constant(a).or_else(|| constant(b))
                .is_some_and(|mask| mask & 3 == 0),
            ("v_mov_b32" | "s_mov_b32", [a]) => constant(a).map_or_else(
                || value(inst, a).is_some_and(|v| self.dword_aligned(v, active, depth)),
                |n| n & 3 == 0,
            ),
            _ => false,
        }
    }

    /// The largest unsigned value `v` takes in any lane that defines it.
    pub(crate) fn unsigned_upper_bound(&mut self, value: Value) -> Option<u32> {
        let hi = self.hi(value);
        (hi != u32::MAX).then_some(hi)
    }

    /// 只在此指令的活动 lane 使用控制流和 EXEC 的比较条件，不污染全局值界。
    pub(crate) fn unsigned_upper_bound_at(&mut self, value: Value, (block, inst): (usize, usize))
        -> Option<u32>
    {
        let reach = (self.st.reach[block], true);
        let guard = self.st.guards[block][inst];
        let facts = [reach, (guard.unwrap_or(Bool::TRUE), true)];
        let local = self.fact_hi(value, &facts[..if guard.is_some() { 2 } else { 1 }], true);
        let hi = self.hi(value).min(local);
        (hi != u32::MAX).then_some(hi)
    }

    /// 诊断仅暴露产生值的指令类别和该位置是否有 EXEC 守卫，不输出指令字节。
    pub(crate) fn value_source_at(&self, v: Value, (b, i): (usize, usize)) -> (&'static str, bool) {
        let root = self.plain_copy_root(v);
        let producer = match self.defs.get(&root).copied() {
            Some(Def::Inst(b, i)) => self.p.blocks[b].insts[i].name(),
            Some(Def::Phi(..)) => "phi",
            None => "input",
        };
        (producer, self.st.guards[b][i].is_some())
    }

    fn plain_copy_root(&self, mut v: Value) -> Value {
        for _ in 0..16 {
            let Some(Def::Inst(b, i)) = self.defs.get(&v).copied() else { break; };
            let inst = &self.p.blocks[b].insts[i];
            let source = match (&inst.fields, inst.name()) {
                (Fields::Valu(f), "v_mov_b32") if self.st.guards[b][i].is_none()
                    && f.sdwa.is_none() && f.dpp.is_none() && !f.clamp && f.omod == 0
                    && f.neg | f.abs | f.opsel == 0 => f.src.as_slice(),
                (Fields::Sop { src, .. }, "s_mov_b32") => src.as_slice(),
                _ => break,
            };
            let [op] = source else { break; };
            if !dst(inst).is_some_and(|reg| inst.write(reg) == Some(v)) { break; }
            let Some(next) = value(inst, op) else { break; };
            // 不跨越 phi 或条件写入，也不跨线程采用 readfirstlane 的值。
            if dst(inst).is_some_and(|reg| matches!(reg, Reg::S(_)))
                && matches!(op, Operand::Vector { .. })
            {
                break;
            }
            v = next;
        }
        v
    }

    fn hi(&mut self, v: Value) -> u32 {
        if let Some(&h) = self.memo.get(&v) {
            return h;
        }
        if !self.busy.insert(v) {
            return u32::MAX;
        }
        let p = self.p;
        let h = match self.defs.get(&v).copied() {
            Some(Def::Phi(b, i)) => {
                let block = &p.blocks[b];
                let mut h = 0;
                for (k, &pred) in block.preds.iter().enumerate() {
                    h = h.max(self.edge_hi(block.phis[i].args[k], pred, b));
                }
                h
            }
            Some(Def::Inst(b, i)) => {
                let inst = &p.blocks[b].insts[i];
                let reg = inst.writes.iter().find(|w| w.1 == v).map(|w| w.0);
                // Under a guard EXEC may be empty there, and the first lane
                // then is lane 0, which may not run there.
                let lane0 = inst.name() == "v_readfirstlane_b32" && self.st.guards[b][i].is_some();
                let h = if reg.is_some() && reg == dst(inst) && !lane0 {
                    self.op_hi(inst)
                } else {
                    u32::MAX
                };
                match (reg, self.st.guards[b][i]) {
                    (Some(r @ Reg::V(_)), Some(_)) => {
                        h.max(inst.read(r).map_or(u32::MAX, |o| self.hi(o)))
                    }
                    _ => h,
                }
            }
            None => u32::MAX,
        };
        self.busy.remove(&v);
        self.memo.insert(v, h);
        h
    }

    /// The largest value `a` has in lanes going from `pred` to `to`.
    fn edge_hi(&mut self, a: Value, pred: usize, to: usize) -> u32 {
        let mut facts = vec![(self.st.reach[pred], true)];
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = self.st.exits[pred]
            && taken != not_taken
        {
            facts.push((when, taken == to));
        }
        let best = self.fact_hi(a, &facts, false);
        if best == u32::MAX { self.hi(a) } else { best }
    }

    fn fact_hi(&mut self, a: Value, facts: &[(Bool, bool)], follow_copies: bool) -> u32 {
        let mut atoms: Vec<Atom> = facts
            .iter()
            .flat_map(|&(f, _)| self.st.bools.support(f))
            .collect();
        atoms.dedup();
        let mut best = u32::MAX;
        for atom in atoms {
            let (Atom::Lane(c) | Atom::Uniform(c)) = atom else {
                continue;
            };
            if let Some(holds) = self.st.bools.decides(facts, atom)
                && let Some(h) = self.compare_bound(c, a, holds, follow_copies)
            {
                best = best.min(h);
            }
        }
        best
    }

    /// What an unsigned compare whose result `c` is `holds` says of the
    /// largest value of its operand `a`.
    fn compare_bound(&mut self, c: Value, a: Value, holds: bool, follow_copies: bool) -> Option<u32> {
        let Some(&Def::Inst(b, i)) = self.defs.get(&c) else {
            return None;
        };
        let p = self.p;
        let inst = &p.blocks[b].insts[i];
        let name = inst.name();
        let (op, src) = match &inst.fields {
            Fields::Valu(v) if v.sdwa.is_none() && v.dpp.is_none()
                && !v.clamp && v.omod == 0 && v.neg | v.abs | v.opsel == 0 => (
                name.strip_prefix("v_cmpx_")
                    .or_else(|| name.strip_prefix("v_cmp_"))?,
                &v.src,
            ),
            Fields::Sop { src, .. } if inst.write(Reg::Scc) == Some(c) => {
                (name.strip_prefix("s_cmp_")?, src)
            }
            _ => return None,
        };
        let op = op.strip_suffix("_u32")?;
        let [x, y] = &src[..] else { return None };
        let (x, y) = (value(inst, x), value(inst, y));
        let same = |v: Option<Value>| v.is_some_and(|v| {
            if follow_copies { self.plain_copy_root(v) == self.plain_copy_root(a) } else { v == a }
        });
        // 比较与访问必须是同一 SSA 值，或无修饰、无条件的逐位复制。
        let (op, other) = match (same(x), same(y)) {
            (true, false) => (op, &src[1]),
            (false, true) => (
                match op {
                    "lt" => "gt",
                    "le" => "ge",
                    "gt" => "lt",
                    "ge" => "le",
                    o => o,
                },
                &src[0],
            ),
            _ => return None,
        };
        let below = match (op, holds) {
            ("lt", true) | ("ge", false) => true,
            ("le", true) | ("gt", false) | ("eq", true) => false,
            _ => return None,
        };
        let h = self.operand_hi(inst, other);
        Some(if below { h.saturating_sub(1) } else { h })
    }

    /// The largest result of `inst`, from its operands.
    fn op_hi(&mut self, inst: &Inst) -> u32 {
        let src = match &inst.fields {
            Fields::Valu(v)
                if v.sdwa.is_none()
                    && v.dpp.is_none()
                    && !v.clamp
                    && v.omod == 0
                    && v.neg | v.abs | v.opsel == 0 =>
            {
                &v.src
            }
            Fields::Sop { src, .. } => src,
            _ => return u32::MAX,
        };
        let s: Vec<u32> = src.iter().map(|o| self.operand_hi(inst, o)).collect();
        match (inst.name(), &s[..]) {
            ("s_mov_b32" | "v_mov_b32" | "v_readfirstlane_b32", &[x]) => x,
            ("s_min_u32" | "v_min_u32" | "s_and_b32" | "v_and_b32", &[x, y]) => x.min(y),
            ("s_max_u32" | "v_max_u32", &[x, y]) => x.max(y),
            // The median is at most the larger of any two operands.
            ("v_med3_u32", &[x, y, z]) => x.max(y).min(x.max(z)).min(y.max(z)),
            // A sum that may wrap is unbounded.
            ("s_add_u32" | "s_add_i32" | "v_add_nc_u32", &[x, y]) => x.saturating_add(y),
            ("s_mul_i32" | "v_mul_lo_u32", &[x, y]) => x.saturating_mul(y),
            ("s_lshl_b32", &[x, shift]) | ("v_lshlrev_b32", &[shift, x]) if shift < 32 =>
                x.saturating_mul(1u32 << shift),
            ("v_lshl_add_u32", &[x, shift, y]) if shift < 32 =>
                x.saturating_mul(1u32 << shift).saturating_add(y),
            ("v_add_lshl_u32", &[x, y, shift]) if shift < 32 =>
                x.saturating_add(y).saturating_mul(1u32 << shift),
            ("v_mad_u32_u24", &[x, y, z]) =>
                x.min(0xff_ffff).saturating_mul(y.min(0xff_ffff)).saturating_add(z),
            _ => u32::MAX,
        }
    }

    fn operand_hi(&mut self, inst: &Inst, op: &Operand) -> u32 {
        match *op {
            Operand::Const(c @ 128..=192) => u32::from(c - 128),
            Operand::Literal(x) => x,
            _ => value(inst, op).map_or(u32::MAX, |v| self.hi(v)),
        }
    }
}

/// The value a one-dword register operand reads.
fn value(inst: &Inst, op: &Operand) -> Option<Value> {
    match *op {
        Operand::Vector { reg, dwords: 1 } => inst.read(Reg::V(reg)),
        Operand::Scalar { code, dwords: 1 } if code != EXEC => inst.read(Reg::S(code)),
        _ => None,
    }
}

/// The register an instruction's one-dword result goes to.
fn dst(inst: &Inst) -> Option<Reg> {
    match &inst.fields {
        Fields::Valu(v) => match v.dst? {
            Operand::Vector { reg, dwords: 1 } => Some(Reg::V(reg)),
            Operand::Scalar { code, dwords: 1 } => Some(Reg::S(code)),
            _ => None,
        },
        Fields::Sop {
            dst: Some(Operand::Scalar { code, dwords: 1 }),
            ..
        } => Some(Reg::S(*code)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;
    const V: u32 = 256;

    const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
        0x7E00_0000 | vdst << 17 | op << 9 | src0
    }

    const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
        op << 25 | vdst << 17 | vsrc1 << 9 | src0
    }

    /// `s_mov_b32 m0, s0`.
    const M0_S0: u32 = 0xBEFC_0300;

    fn build_words(words: &[u32]) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let (p, st) = build(&code, false, 16).unwrap();
        st.unwrap();
        p
    }

    fn vgprs(regs: &[(Reg, Value)]) -> Vec<u8> {
        regs.iter()
            .filter_map(|r| match r.0 {
                Reg::V(v) => Some(v),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn hardware_bounds_apply_only_to_the_entry_value() {
        let p = build_words(&[
            vop1(0x01, 1, V),           // v1 读取入口 v0
            vop2(0x25, 0, 129, 0),     // v0 加一并覆盖寄存器
            S_ENDPGM,
        ]);
        let st = structure::structure(&p).unwrap();
        let i = &p.blocks[0].insts;
        let value = i[0].write(Reg::V(1)).unwrap();
        let overwritten = i[1].write(Reg::V(0)).unwrap();
        let mut bounded = Ranges::new(&p, &st);
        bounded.bound_input(Reg::V(0), 7);
        assert_eq!(bounded.unsigned_upper_bound(value), Some(7));
        assert_eq!(bounded.unsigned_upper_bound(overwritten), Some(8));
        assert_eq!(Ranges::new(&p, &st).unsigned_upper_bound(value), None);
    }

    #[test]
    fn lane_compare_limits_only_the_guarded_site_and_matching_value() {
        let p = build_words(&[
            vop1(0x01, 1, V),      // 读取 v0，未缩小 EXEC。
            0x7DA4_0080,           // v_cmpx_eq_u32 exec, 0, v0
            vop1(0x01, 2, V),      // 仅在 v0 等于 0 的 lane 运行。
            vop1(0x01, 3, V + 4),  // v4 未经比较。
            S_ENDPGM,
        ]);
        let st = structure::structure(&p).unwrap();
        let i = &p.blocks[0].insts;
        let v0 = i[0].read(Reg::V(0)).unwrap();
        let unrelated = i[3].read(Reg::V(4)).unwrap();
        let mut range = Ranges::new(&p, &st);
        assert_eq!(range.unsigned_upper_bound_at(v0, (0, 2)), Some(0));
        assert_eq!(range.unsigned_upper_bound_at(v0, (0, 0)), None);
        assert_eq!(range.unsigned_upper_bound_at(unrelated, (0, 3)), None);
        assert_eq!(range.unsigned_upper_bound(v0), None);
    }

    #[test]
    fn compares_follow_plain_copies_but_not_guarded_overwrites() {
        let p = build_words(&[
            vop1(0x01, 1, V),      // v1 是 v0 的无条件复制。
            0x7DA4_0080,           // v_cmpx_eq_u32 exec, 0, v0
            vop1(0x01, 2, V + 1),  // 比较过 v0 就能约束 v1。
            vop1(0x01, 1, V + 4),  // 活动 lane 内 v1 改成未比较的 v4。
            vop1(0x01, 2, V + 1),
            S_ENDPGM,
        ]);
        let st = structure::structure(&p).unwrap();
        let i = &p.blocks[0].insts;
        let copy = i[2].read(Reg::V(1)).unwrap();
        let overwrite = i[4].read(Reg::V(1)).unwrap();
        let mut range = Ranges::new(&p, &st);
        assert_eq!(range.unsigned_upper_bound_at(copy, (0, 2)), Some(0));
        assert_eq!(range.unsigned_upper_bound_at(overwrite, (0, 4)), None);
    }

    #[test]
    fn a_bound_must_survive_what_the_bounded_moves_write() {
        // v1 = v0 & 3, vN = v0 & 1; v[2 + v1] = v7; then v[8 + vN] is read.
        // The first move reaches v2-v5: with N = 4 it may overwrite the
        // second index, which then has no bound (it would, were the first
        // move taken to write only v2); with N = 6 the second reads v8-v9.
        for (n, second) in [
            (4, (8..16).chain([1]).collect::<Vec<u8>>()),
            (6, vec![8, 9, 1]),
        ] {
            let p = build_words(&[
                vop2(0x1B, 1, 131, 0), // v_and_b32 v1, 3, v0
                vop2(0x1B, n, 129, 0), // v_and_b32 vN, 1, v0
                vop1(0x02, 0, V + 1),  // v_readfirstlane_b32 s0, v1
                M0_S0,
                vop1(0x42, 2, V + 7), // v_movreld_b32 v2, v7
                vop1(0x02, 0, V + n), // v_readfirstlane_b32 s0, vN
                M0_S0,
                vop1(0x43, 1, V + 8), // v_movrels_b32 v1, v8
                S_ENDPGM,
            ]);
            let i = &p.blocks[0].insts;
            assert_eq!(vgprs(&i[4].writes), [2, 3, 4, 5]);
            assert_eq!(vgprs(&i[7].reads), second, "index in v{n}");
        }
    }

    #[test]
    fn a_loop_counter_bounds_the_index() {
        // for (v7 = 0; v7 < 4; v7++) v[2 + v7] = v0. Without the bound the
        // move could overwrite its own counter.
        let p = build_words(&[
            vop1(0x01, 7, 128),                      // v_mov_b32 v7, 0
            vop1(0x02, 0, V + 7),                    // v_readfirstlane_b32 s0, v7
            M0_S0,                                   //
            vop1(0x42, 2, V),                        // v_movreld_b32 v2, v0
            vop2(0x25, 7, 129, 7),                   // v_add_nc_u32 v7, 1, v7
            0x7C00_0000 | 0xC4 << 17 | 7 << 9 | 132, // v_cmp_gt_u32 vcc, 4, v7
            0xBF87_FFFA,                             // s_cbranch_vccnz to the readfirstlane
            S_ENDPGM,
        ]);
        let movreld = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.name() == "v_movreld_b32")
            .unwrap();
        assert_eq!(vgprs(&movreld.writes), [2, 3, 4, 5]);
        // A limit of 5 lets the index reach 4: v2-v6.
        let p = build_words(&[
            vop1(0x01, 7, 128),
            vop1(0x02, 0, V + 7),
            M0_S0,
            vop1(0x42, 2, V),
            vop2(0x25, 7, 129, 7),
            0x7C00_0000 | 0xC4 << 17 | 7 << 9 | 133, // v_cmp_gt_u32 vcc, 5, v7
            0xBF87_FFFA,
            S_ENDPGM,
        ]);
        let movreld = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.name() == "v_movreld_b32")
            .unwrap();
        assert_eq!(vgprs(&movreld.writes), [2, 3, 4, 5, 6]);
    }

    #[test]
    fn a_first_lane_where_exec_may_be_empty_is_unbounded() {
        // In lanes with v0 < 2: v1 = v0 & 3; then EXEC = lanes with v0 < 1,
        // possibly none, and m0 = the first lane's v1: with none, lane 0's,
        // which may never have set v1. Read before narrowing EXEC, the index
        // is bounded.
        for (narrow_first, reads) in [
            (true, (8..16).collect::<Vec<u8>>()),
            (false, vec![8, 9, 10, 11]),
        ] {
            let narrow = [
                0x7C00_0000 | 0xC4 << 17 | 129, // v_cmp_gt_u32 vcc, 1, v0
                0xBE83_3C6A,                    // s_and_saveexec_b32 s3, vcc_lo
            ];
            let first = vop1(0x02, 0, V + 1); // v_readfirstlane_b32 s0, v1
            let mut words = vec![
                0x7C00_0000 | 0xC4 << 17 | 130, // v_cmp_gt_u32 vcc, 2, v0
                0xBE82_3C6A,                    // s_and_saveexec_b32 s2, vcc_lo
                0xBF88_0007,                    // s_cbranch_execz to the end
                vop2(0x1B, 1, 131, 0),          // v_and_b32 v1, 3, v0
            ];
            if narrow_first {
                words.extend(narrow);
                words.push(first);
            } else {
                words.push(first);
                words.extend(narrow);
            }
            words.extend([
                M0_S0,
                0xBEFE_0303,          // s_mov_b32 exec_lo, s3
                vop1(0x43, 4, V + 8), // v_movrels_b32 v4, v8
                0xBEFE_0302,          // s_mov_b32 exec_lo, s2
                S_ENDPGM,
            ]);
            let p = build_words(&words);
            let movrels = p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| i.name() == "v_movrels_b32")
                .unwrap();
            assert_eq!(
                vgprs(&movrels.reads)[..reads.len()],
                reads[..],
                "{narrow_first}"
            );
        }
    }
}
