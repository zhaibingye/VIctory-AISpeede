//! Native scalar control for a complete 32-lane compute wave. All lanes
//! visit the same IR blocks. EXEC guards vector writes and memory effects.
//! AMD's *RDNA 2 Instruction Set Architecture*, §§3.3, 3.6 and 4.1,
//! describes the wave's EXEC mask, scalar registers and program flow.

use super::{Attr, Bin, EXEC, Fields, Inst, Lower, Reg, V, Val, Value, Why};

impl Lower<'_> {
    pub(super) fn exec_word(&mut self, inst: &Inst) -> Result<V, Why> {
        self.i32_of(inst.read(Reg::S(EXEC)).ok_or("a missing EXEC operand")?)
    }

    pub(super) fn collect_wave_masks(&mut self, inst: &Inst) -> Result<(), Why> {
        // A VALU compare, cmpx or carry writes the whole scalar mask. Bits
        // for lanes disabled in the incoming EXEC are cleared.
        for &(reg, value) in &inst.writes {
            if let Reg::S(_) = reg
                && let Some(&bit) = self.lane_bits.get(&value)
            {
                let exec = self.exec_word(inst)?;
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                self.s.flags |= super::flags::WAVE_OPS;
                let result = self.s.op(
                    self.cur,
                    "waveActiveBallot",
                    Attr::NoUnwind,
                    four,
                    116,
                    &[bit],
                );
                let low = self.s.m.extract(self.cur, result, 0);
                let word = self.s.m.bin(self.cur, Bin::And, low, exec);
                self.vals.insert(value, Val::I32(word));
            }
        }
        Ok(())
    }

    pub(super) fn update_exec(
        &mut self,
        inst: &Inst,
        dst: Option<Value>,
        scc: Option<Value>,
    ) -> Result<(), Why> {
        let Fields::Sop { src, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let op = name
            .strip_prefix("s_")
            .and_then(|n| n.split_once('_'))
            .map(|(op, _)| op)
            .filter(|_| name.ends_with("_b32"))
            .ok_or("a wide EXEC update")?;
        let previous = self.exec_word(inst)?;
        let mut a = self.operand(inst, src.first().ok_or("a missing EXEC source")?)?;
        let mut b = previous;
        let c = self.cur;
        let ones = self.s.m.i32(u32::MAX);
        if op.ends_with("n1") {
            a = self.s.m.bin(c, Bin::Xor, a, ones);
        }
        if op.ends_with("n2") {
            b = self.s.m.bin(c, Bin::Xor, b, ones);
        }
        let operation = match op {
            "and" | "andn1" | "andn2" | "nand" => Bin::And,
            "or" | "orn1" | "orn2" | "nor" => Bin::Or,
            "xor" | "xnor" => Bin::Xor,
            _ => return Err("an unhandled EXEC update"),
        };
        let mut next = self.s.m.bin(c, operation, a, b);
        if matches!(op, "nand" | "nor" | "xnor") {
            next = self.s.m.bin(c, Bin::Xor, next, ones);
        }
        let saved = if name.contains("_saveexec_") {
            previous
        } else {
            next
        };
        self.set(dst, Val::I32(saved));
        self.set(inst.write(Reg::S(EXEC)), Val::I32(next));
        let zero = self.s.m.i32(0);
        let nonzero = self.s.m.cmp(c, super::pred::NE, next, zero);
        self.set(scc, Val::Bit(nonzero));
        Ok(())
    }

    /// Expand each nonempty four-bit group to a full quad.
    pub(super) fn quad_mask_word(&mut self, value: V) -> V {
        let c = self.cur;
        let one = self.s.m.i32(1);
        let two = self.s.m.i32(2);
        let adjacent = self.s.m.bin(c, Bin::LShr, value, one);
        let pairs = self.s.m.bin(c, Bin::Or, value, adjacent);
        let adjacent = self.s.m.bin(c, Bin::LShr, pairs, two);
        let quads = self.s.m.bin(c, Bin::Or, pairs, adjacent);
        let low_bits = self.s.m.i32(0x1111_1111);
        let low = self.s.m.bin(c, Bin::And, quads, low_bits);
        let nibble = self.s.m.i32(15);
        self.s.m.bin(c, Bin::Mul, low, nibble)
    }

    /// Each destination lane takes one nibble of the two selector words.
    /// PERMLANEX16 reads the other 16-lane row of the wave.
    pub(super) fn permute_row(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(f) = &inst.fields else {
            unreachable!()
        };
        if f.opsel & !3 != 0 || f.abs != 0 || f.neg != 0 || f.omod != 0 || f.clamp {
            return Err("a lane permutation with modifiers");
        }
        let (data, low, high) = (self.src(inst, 0)?, self.src(inst, 1)?, self.src(inst, 2)?);
        let c = self.cur;
        let lane = self.lane_index();
        let (k2, k7, k8, k15, k16, zero) = (
            self.s.m.i32(2),
            self.s.m.i32(7),
            self.s.m.i32(8),
            self.s.m.i32(15),
            self.s.m.i32(16),
            self.s.m.i32(0),
        );
        let position = self.s.m.bin(c, Bin::And, lane, k7);
        let shift = self.s.m.bin(c, Bin::Shl, position, k2);
        let half = self.s.m.bin(c, Bin::And, lane, k8);
        let first = self.s.m.cmp(c, super::pred::EQ, half, zero);
        let selectors = self.s.m.select(c, first, low, high);
        let nibble = self.s.m.bin(c, Bin::LShr, selectors, shift);
        let column = self.s.m.bin(c, Bin::And, nibble, k15);
        let mut row = self.s.m.bin(c, Bin::And, lane, k16);
        if inst.name() == "v_permlanex16_b32" {
            row = self.s.m.bin(c, Bin::Xor, row, k16);
        }
        let source = self.s.m.bin(c, Bin::Or, row, column);
        let mut value = self.read_lane(data, source);
        let mut enabled = guard;
        if f.opsel & 1 == 0 {
            let exec = self.exec_word(inst)?;
            let active = self.lane_bit_of(c, source, exec, zero);
            if f.opsel & 2 != 0 {
                value = self.s.m.select(c, active, value, zero);
            } else {
                enabled = Some(guard.map_or(active, |g| self.s.m.bin(c, Bin::And, g, active)));
            }
        }
        self.write_i(inst, value, enabled)
    }

    /// Insert at the scalar load so all later users, including phi edges,
    /// share the same descriptor index and the load dominates every use.
    pub(super) fn descriptor_base_word(&mut self, value: Value) -> Result<V, Why> {
        let (site, first, block) = self.descriptor_bases[&value];
        let saved = (self.cur_ir, self.cur);
        (self.cur_ir, self.cur) = (site.0, block);
        let index = self.heap_index(site, first, false);
        (self.cur_ir, self.cur) = saved;
        let index = index?;
        let views = self.s.view_addresses_handle();
        let stride = self.s.m.i32(crate::emit::VIEW_ADDRESS_BYTES);
        let record = self.s.m.bin(block, Bin::Mul, index, stride);
        self.cur = block;
        let word = self.raw_dword(views, record);
        self.cur = saved.1;
        self.vals.insert(value, Val::I32(word));
        Ok(word)
    }
}
