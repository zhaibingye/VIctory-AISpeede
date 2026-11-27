//! Bounded probe of the finalized emitter CFG, not of the guest instructions.

use super::{Cast, Const, Module, Op, Term, Ty, TypeDef, V, successors};
use crate::emit::PixelAlphaProbe;

#[cfg(test)]
mod tests;

const MAX_BLOCKS: usize = 8_192;
const MAX_INSTS: usize = 262_144;
const MAX_FUNCS: usize = 1_024;
const MAX_VALUE_NODES: usize = u64::BITS as usize;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Alpha {
    Unknown, // Unwritten, unproven, or disagreement at a join.
    Opaque,
    AtMostHalf,
}

fn merge(into: &mut Option<Alpha>, incoming: Alpha) {
    *into = Some(match *into {
        Some(old) if old != incoming => Alpha::Unknown,
        _ => incoming,
    });
}

struct Flow {
    postorder: Vec<usize>,
    preds: Vec<Vec<usize>>,
    phi_blocks: Vec<Option<usize>>,
}

impl Module {
    /// Analyze the emitter's own SSA (not arbitrary DXIL) for one f32 output
    /// component. The result concerns surviving returns, not pixel coverage.
    pub(in crate::emit) fn probe_pixel_alpha(&self, element: u32, column: u8) -> PixelAlphaProbe {
        self.pixel_alpha(element, column)
            .unwrap_or(PixelAlphaProbe::Unknown)
    }

    fn pixel_alpha(&self, element: u32, column: u8) -> Option<PixelAlphaProbe> {
        if column > 3
            || self.blocks.len() > MAX_BLOCKS
            || self.insts.len() > MAX_INSTS
            || self.funcs.len() > MAX_FUNCS
        {
            return None;
        }
        let mut bodies = self.funcs.iter().filter(|f| f.body);
        let main = bodies.next()?;
        if main.name != "main"
            || bodies.next().is_some()
            || !matches!(self.types.get(main.ty.0 as usize), Some(TypeDef::Func(ret, params))
                if self.types.get(ret.0 as usize) == Some(&TypeDef::Void) && params.is_empty())
        {
            return None;
        }
        let flow = self.alpha_flow()?;
        // One joined last-write state per block suffices: a store replaces
        // that state, and the probe never uses it to decide reachability.
        let mut entries = vec![None; self.blocks.len()];
        entries[0] = Some(Alpha::Unknown);
        let mut returns = None;
        let mut discarded = false;
        'block: for &b in flow.postorder.iter().rev() {
            let Some(mut alpha) = entries[b] else {
                continue;
            };
            let block = &self.blocks[b];
            let mut last_store = None;
            for &i in &block.body {
                let inst = self.insts.get(i as usize)?;
                let Op::Call(f, args) = &inst.op else {
                    continue;
                };
                let func = self.funcs.get(*f as usize)?;
                let name = func.name.strip_prefix("dx.op.")?;
                if func.body {
                    return None;
                }
                let (&opcode, _) = args.split_first()?;
                let opcode = self.alpha_i32(opcode)?;
                match (name, opcode) {
                    ("storeOutput.f32" | "storeOutput.i32", 5) => {
                        let [_, id, row, col, value] = args[..] else {
                            return None;
                        };
                        let value_type = if name == "storeOutput.f32" {
                            TypeDef::Float
                        } else {
                            TypeDef::Int(32)
                        };
                        self.alpha_call_shape(
                            func.ty,
                            inst.ty,
                            args,
                            &[
                                TypeDef::Int(32),
                                TypeDef::Int(32),
                                TypeDef::Int(32),
                                TypeDef::Int(8),
                                value_type,
                            ],
                        )?;
                        let id = self.alpha_i32(id)?;
                        let row = self.alpha_i32(row)?;
                        let col = self.alpha_int(col, 8)?;
                        if !(0..=3).contains(&col) {
                            return None;
                        }
                        if (id, row, col) == (element, 0, i64::from(column)) {
                            if name != "storeOutput.f32" {
                                return None;
                            }
                            last_store = Some(value);
                        }
                    }
                    ("discard", 82) => {
                        let [_, cond] = args[..] else { return None };
                        self.alpha_call_shape(
                            func.ty,
                            inst.ty,
                            args,
                            &[TypeDef::Int(32), TypeDef::Int(1)],
                        )?;
                        if self.alpha_condition(cond)? == Some(true) {
                            discarded = true;
                            continue 'block;
                        }
                        // A dynamic discard can still leave this path alive.
                    }
                    (_, 5 | 82) => return None,
                    (name, _) if name.starts_with("storeOutput") || name.starts_with("discard") => {
                        return None;
                    }
                    _ => {} // Other DXIL operations do not overwrite pixel outputs.
                }
            }
            if let Some(value) = last_store {
                let mut budget = MAX_VALUE_NODES;
                alpha = self
                    .alpha_value(value, &flow, &mut budget)
                    .unwrap_or(Alpha::Unknown);
            }
            match block.term? {
                Term::Ret => merge(&mut returns, alpha),
                Term::Br(next) => merge(&mut entries[next], alpha),
                Term::CondBr(cond, t, f) => match self.alpha_condition(cond)? {
                    Some(true) => merge(&mut entries[t], alpha),
                    Some(false) => merge(&mut entries[f], alpha),
                    None => {
                        merge(&mut entries[t], alpha);
                        merge(&mut entries[f], alpha);
                    }
                },
            }
        }
        Some(match returns {
            Some(Alpha::Opaque) => PixelAlphaProbe::OpaqueOnSurvivingReturn,
            Some(Alpha::AtMostHalf) => PixelAlphaProbe::AtMostHalfOnSurvivingReturn,
            None if discarded => PixelAlphaProbe::NoSurvivingReturn,
            _ => PixelAlphaProbe::Unknown,
        })
    }

    fn alpha_flow(&self) -> Option<Flow> {
        let mut flow = Flow {
            postorder: Vec::new(),
            preds: vec![Vec::new(); self.blocks.len()],
            phi_blocks: vec![None; self.insts.len()],
        };
        // Use the same structural successors as write(). The serializer
        // omits unreachable blocks and unread phis, and only folds empty
        // latches on back edges. Acyclic CFGs have no such folds; rejecting
        // cycles also rules out mistaking a nonterminating path for discard.
        let mut marks = vec![0; self.blocks.len()]; // unseen, visiting, done
        let mut stack = vec![(0, false)];
        while let Some((b, finish)) = stack.pop() {
            let mark = marks.get_mut(b)?;
            if finish {
                *mark = 2;
                flow.postorder.push(b);
                continue;
            }
            match *mark {
                2 => continue,
                1 => return None,
                _ => *mark = 1,
            }
            let block = &self.blocks[b];
            if let Term::CondBr(c, ..) = block.term? {
                self.alpha_condition(c)?;
            }
            for &i in &block.phis {
                if !matches!(self.insts.get(i as usize)?.op, Op::Phi(_))
                    || flow.phi_blocks[i as usize].replace(b).is_some()
                {
                    return None;
                }
            }
            stack.push((b, true));
            for &next in successors(block.term).iter() {
                flow.preds.get_mut(next)?.push(b);
                stack.push((next, false));
            }
        }
        Some(flow)
    }

    fn alpha_ty(&self, v: V) -> Option<Ty> {
        Some(match v {
            V::Const(i) => self.consts.get(i as usize)?.0,
            V::Inst(i) => self.insts.get(i as usize)?.ty,
            V::Global(i) => self.globals.get(i as usize)?.ptr,
            V::Func(_) => return None,
        })
    }

    fn alpha_type(&self, v: V) -> Option<&TypeDef> {
        self.types.get(self.alpha_ty(v)?.0 as usize)
    }

    fn alpha_int(&self, v: V, width: u32) -> Option<i64> {
        let V::Const(i) = v else { return None };
        let (ty, Const::Int(n)) = self.consts.get(i as usize)? else {
            return None;
        };
        (self.types.get(ty.0 as usize) == Some(&TypeDef::Int(width))).then_some(*n)
    }

    fn alpha_i32(&self, v: V) -> Option<u32> {
        let n = self.alpha_int(v, 32)?;
        u32::try_from(n)
            .ok()
            .or_else(|| i32::try_from(n).ok().map(|n| n as u32))
    }

    /// Outer `None` is an invalid condition, inner `None` a possible either-way branch.
    fn alpha_condition(&self, v: V) -> Option<Option<bool>> {
        if self.alpha_type(v) != Some(&TypeDef::Int(1)) {
            return None;
        }
        let V::Const(i) = v else { return Some(None) };
        match self.consts.get(i as usize)?.1 {
            Const::Int(0) => Some(Some(false)),
            Const::Int(-1 | 1) => Some(Some(true)),
            Const::Undef => Some(None),
            _ => None,
        }
    }

    fn alpha_call_shape(&self, decl: Ty, ret: Ty, args: &[V], types: &[TypeDef]) -> Option<()> {
        let TypeDef::Func(result, params) = self.types.get(decl.0 as usize)? else {
            return None;
        };
        if *result != ret
            || self.types.get(ret.0 as usize) != Some(&TypeDef::Void)
            || params.len() != args.len()
            || params.len() != types.len()
        {
            return None;
        }
        for ((&ty, &v), expected) in params.iter().zip(args).zip(types) {
            if self.alpha_ty(v) != Some(ty) || self.types.get(ty.0 as usize) != Some(expected) {
                return None;
            }
        }
        Some(())
    }

    /// Prove a class of f32 bits, including through scalar i32 bitcasts.
    /// Joins must agree even on incoming edges the CFG probe may prune.
    fn alpha_value(&self, v: V, flow: &Flow, budget: &mut usize) -> Option<Alpha> {
        *budget = budget.checked_sub(1)?;
        let ty = self.alpha_type(v)?;
        if !matches!(ty, TypeDef::Float | TypeDef::Int(32)) {
            return None;
        }
        match v {
            V::Const(i) => {
                let bits = match &self.consts[i as usize].1 {
                    Const::Float(bits) if *ty == TypeDef::Float => *bits,
                    Const::Int(_) if *ty == TypeDef::Int(32) => self.alpha_i32(v)?,
                    _ => return None,
                };
                let value = f32::from_bits(bits);
                if bits == 1.0f32.to_bits() {
                    Some(Alpha::Opaque)
                } else if value.is_finite() && value <= 0.5 {
                    Some(Alpha::AtMostHalf)
                } else {
                    None
                }
            }
            V::Inst(i) => {
                let check = |v| (self.alpha_type(v) == Some(ty)).then_some(v);
                match &self.insts.get(i as usize)?.op {
                    Op::Cast(Cast::Bitcast, src)
                        if matches!(
                            (ty, self.alpha_type(*src)?),
                            (TypeDef::Float, TypeDef::Int(32)) | (TypeDef::Int(32), TypeDef::Float)
                        ) =>
                    {
                        self.alpha_value(*src, flow, budget)
                    }
                    Op::Select(c, t, f) => {
                        let t = check(*t)?;
                        let f = check(*f)?;
                        match self.alpha_condition(*c)? {
                            Some(true) => self.alpha_value(t, flow, budget),
                            Some(false) => self.alpha_value(f, flow, budget),
                            None => {
                                let t = self.alpha_value(t, flow, budget)?;
                                let f = self.alpha_value(f, flow, budget)?;
                                (t == f).then_some(t)
                            }
                        }
                    }
                    Op::Phi(incoming) => {
                        let preds = &flow.preds[flow.phi_blocks[i as usize]?];
                        if incoming.len() != preds.len() || incoming.len() > *budget {
                            return None;
                        }
                        let mut seen = 0u64;
                        let mut joined = None;
                        for &(v, pred) in incoming {
                            let bit = 1u64 << preds.iter().position(|&p| p == pred)?;
                            if seen & bit != 0 {
                                return None;
                            }
                            seen |= bit;
                            let v = self.alpha_value(check(v)?, flow, budget)?;
                            if joined.is_some_and(|old| old != v) {
                                return None;
                            }
                            joined = Some(v);
                        }
                        joined
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
