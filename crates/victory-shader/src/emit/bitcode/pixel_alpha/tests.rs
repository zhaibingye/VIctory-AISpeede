use super::*;
use crate::emit::bitcode::{Attr, pred};
use PixelAlphaProbe::{
    AtMostHalfOnSurvivingReturn as Half, NoSurvivingReturn as Dead,
    OpaqueOnSurvivingReturn as Opaque, Unknown,
};

fn module() -> Module {
    let mut m = Module::default();
    let void = m.void();
    let ty = m.func_ty(void, vec![]);
    m.ptr(ty);
    m.function("main", ty, None, true);
    assert_eq!(m.new_block(), 0);
    m
}

fn dx(m: &mut Module, b: usize, name: &str, ret: Ty, opcode: u32, args: &[V]) -> V {
    let opcode = m.i32(opcode);
    let args: Vec<_> = std::iter::once(opcode)
        .chain(args.iter().copied())
        .collect();
    let ty = m.func_ty(ret, args.iter().map(|&v| m.type_of(v)).collect());
    m.ptr(ty);
    let f = m.function(name, ty, Some(Attr::NoUnwind), false);
    m.call(b, f, args)
}

fn write(m: &mut Module, b: usize, v: V) {
    let id = m.i32(0);
    let row = m.i32(0);
    let col = m.i8(3);
    let void = m.void();
    dx(m, b, "dx.op.storeOutput.f32", void, 5, &[id, row, col, v]);
}

fn literal(m: &mut Module, b: usize, v: f32) {
    let v = m.f32(v.to_bits());
    write(m, b, v);
}

fn discard(m: &mut Module, b: usize, cond: V) {
    let void = m.void();
    dx(m, b, "dx.op.discard", void, 82, &[cond]);
}

fn runtime_alpha(m: &mut Module) -> V {
    let ty = m.float();
    let row = m.i32(0);
    let col = m.i8(0);
    let i32t = m.int(32);
    let index = m.undef(i32t);
    dx(m, 0, "dx.op.loadInput.f32", ty, 4, &[row, row, col, index])
}

fn runtime_bool(m: &mut Module) -> V {
    let a = runtime_alpha(m);
    let zero = m.f32(0);
    m.cmp(0, pred::OLT, a, zero)
}

fn probe(m: &Module) -> PixelAlphaProbe {
    m.probe_pixel_alpha(0, 3)
}

#[test]
fn literal_sign_reversed_and_runtime_controls() {
    for (value, result) in [
        (0.5, Half),
        (1.0, Opaque),
        (0.75, Unknown),
        (f32::NAN, Unknown),
        (f32::NEG_INFINITY, Unknown),
    ] {
        let mut m = module();
        literal(&mut m, 0, value);
        m.terminate(0, Term::Ret);
        assert_eq!(probe(&m), result);
        assert_eq!(m.probe_pixel_alpha(1, 3), Unknown);
        assert_eq!(m.probe_pixel_alpha(0, 2), Unknown);
    }
    let mut m = module();
    let a = runtime_alpha(&mut m);
    write(&mut m, 0, a);
    m.terminate(0, Term::Ret);
    assert_eq!(probe(&m), Unknown);
}

#[test]
fn last_write_wins_on_every_surviving_return() {
    for (early, late, result) in [(0.5, 1.0f32, Opaque), (1.0, 0.5, Half)] {
        let mut m = module();
        let next = m.new_block();
        literal(&mut m, 0, early);
        m.terminate(0, Term::Br(next));
        let a = m.i32(late.to_bits());
        let ty = m.float();
        let a = m.cast(next, Cast::Bitcast, a, ty);
        write(&mut m, next, a);
        m.terminate(next, Term::Ret);
        assert_eq!(probe(&m), result);
    }
    let mut m = module();
    let a = runtime_alpha(&mut m);
    write(&mut m, 0, a);
    literal(&mut m, 0, 0.5);
    m.terminate(0, Term::Ret);
    assert_eq!(probe(&m), Half);
    write(&mut m, 0, a);
    assert_eq!(probe(&m), Unknown);
}

#[test]
fn bypass_and_discard_paths() {
    let mut m = module();
    let yes = m.new_block();
    let no = m.new_block();
    let runtime = runtime_bool(&mut m);
    m.terminate(0, Term::CondBr(runtime, yes, no));
    literal(&mut m, yes, 0.5);
    m.terminate(yes, Term::Ret);
    m.terminate(no, Term::Ret);
    assert_eq!(probe(&m), Unknown); // one return does not write alpha
    discard(&mut m, no, runtime);
    assert_eq!(probe(&m), Unknown); // a dynamic discard is not a proof of death
    let keep = m.bool(false);
    discard(&mut m, no, keep);
    assert_eq!(probe(&m), Unknown);
    let kill = m.bool(true);
    discard(&mut m, no, kill);
    literal(&mut m, no, 1.0);
    assert_eq!(probe(&m), Half); // later instructions must not revive the lane
    discard(&mut m, yes, kill);
    assert_eq!(probe(&m), Dead);
}

#[test]
fn branches_prune_only_literal_conditions_and_overwrites_clear_disagreement() {
    let mut m = module();
    let yes = m.new_block();
    let no = m.new_block();
    let done = m.new_block();
    let runtime = runtime_bool(&mut m);
    m.terminate(0, Term::CondBr(runtime, yes, no));
    literal(&mut m, yes, 0.5);
    m.terminate(yes, Term::Br(done));
    literal(&mut m, no, 1.0);
    m.terminate(no, Term::Br(done));
    m.terminate(done, Term::Ret);
    assert_eq!(probe(&m), Unknown);
    let c = m.bool(true);
    m.terminate(0, Term::CondBr(c, yes, no));
    assert_eq!(probe(&m), Half);
    let c = m.bool(false);
    m.terminate(0, Term::CondBr(c, yes, no));
    assert_eq!(probe(&m), Opaque);
    m.terminate(0, Term::CondBr(runtime, yes, no));
    literal(&mut m, done, 0.5);
    assert_eq!(probe(&m), Half);
}

#[test]
fn phi_and_select_prove_only_agreeing_classes() {
    let mut m = module();
    let left = m.new_block();
    let right = m.new_block();
    let join = m.new_block();
    let cond = runtime_bool(&mut m);
    m.terminate(0, Term::CondBr(cond, left, right));
    m.terminate(left, Term::Br(join));
    m.terminate(right, Term::Br(join));
    m.terminate(join, Term::Ret);
    let ty = m.int(32);
    let phi = m.phi(join, ty);
    let a = m.i32(0.5f32.to_bits());
    let b = m.i32(0.25f32.to_bits());
    let one = m.i32(1.0f32.to_bits());
    let selected = m.select(join, cond, a, phi);
    let ft = m.float();
    let alpha = m.cast(join, Cast::Bitcast, selected, ft);
    write(&mut m, join, alpha);
    m.set_phi(phi, vec![(a, left), (b, right)]);
    assert_eq!(probe(&m), Half);
    m.set_phi(phi, vec![(a, left), (one, right)]);
    assert_eq!(probe(&m), Unknown);
    m.set_phi(phi, vec![(a, left), (b, left)]);
    assert_eq!(probe(&m), Unknown); // malformed phi cannot prove alpha
}

#[test]
fn requires_exact_store_output_and_literal_address() {
    for (name, opcode, dynamic_col) in [
        ("dx.op.storeOutput.f32", 6, false),
        ("dx.op.storeOutput.i32", 5, false),
        ("dx.op.storeOutput.f32", 5, true),
        ("dx.op.unknown.f32", 5, false),
    ] {
        let mut m = module();
        let id = m.i32(0);
        let col = if dynamic_col {
            let i8 = m.int(8);
            m.undef(i8)
        } else {
            m.i8(3)
        };
        let v = if name.ends_with(".i32") {
            m.i32(0)
        } else {
            m.f32(0)
        };
        let void = m.void();
        dx(&mut m, 0, name, void, opcode, &[id, id, col, v]);
        literal(&mut m, 0, 0.5);
        m.terminate(0, Term::Ret);
        assert_eq!(probe(&m), Unknown);
    }
    let mut m = module();
    literal(&mut m, 0, 0.5);
    let id = m.i32(0);
    let col = m.i8(2);
    let void = m.void();
    dx(
        &mut m,
        0,
        "dx.op.storeOutput.i32",
        void,
        5,
        &[id, id, col, id],
    );
    m.terminate(0, Term::Ret);
    assert_eq!(probe(&m), Half); // exact but disjoint integer store
}

#[test]
fn absent_malformed_or_cyclic_flow_is_unknown() {
    assert_eq!(probe(&Module::default()), Unknown);
    let mut m = module();
    let b = m.new_block();
    literal(&mut m, 0, 1.0);
    assert_eq!(probe(&m), Unknown);
    m.terminate(0, Term::Br(b));
    assert_eq!(probe(&m), Unknown);
    m.terminate(b, Term::Br(usize::MAX));
    assert_eq!(probe(&m), Unknown);
    m.terminate(b, Term::Br(b));
    assert_eq!(probe(&m), Unknown);
    let done = m.new_block();
    let cond = runtime_bool(&mut m);
    let kill = m.bool(true);
    m.terminate(0, Term::CondBr(cond, done, b));
    discard(&mut m, done, kill);
    m.terminate(done, Term::Ret);
    assert_eq!(probe(&m), Unknown); // one discard does not rule out infinite paths
    let invalid = m.i32(1);
    m.terminate(0, Term::CondBr(invalid, done, done));
    assert_eq!(probe(&m), Unknown);
}

#[test]
fn value_proof_has_a_work_budget() {
    let mut m = module();
    let mut a = m.f32(0.5f32.to_bits());
    let it = m.int(32);
    let ft = m.float();
    for _ in 0..MAX_VALUE_NODES {
        a = m.cast(0, Cast::Bitcast, a, it);
        a = m.cast(0, Cast::Bitcast, a, ft);
    }
    write(&mut m, 0, a);
    m.terminate(0, Term::Ret);
    assert_eq!(probe(&m), Unknown);
}
