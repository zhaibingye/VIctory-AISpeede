//! Shader dependency admission for the closed page renderer. An access
//! with no lowering is rejected, not filled with the legacy root views.
use victory_shader::emit::{HostShader, UserData};
use victory_shader::{RootType, UsedResource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnloweredShader {
    GraphicsState,
    DynamicAddress,
    DescriptorHeap,
    RootView,
    DescriptorTable,
    UnknownUserData,
    MissingConstant { slot: u8, dword: u32 },
    ConstantLayout,
}

pub(crate) struct RootWords {
    pub(crate) parameter: u32,
    pub(crate) words: Vec<u32>,
}

pub(crate) fn constants_only(
    shader: &HostShader,
    used: &[UsedResource],
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
) -> Result<Option<RootWords>, UnloweredShader> {
    if shader.memory || shader.memory_writes || shader.view_addresses {
        return Err(UnloweredShader::DynamicAddress);
    }
    if shader.heap || shader.heap_writes || shader.view_marks {
        return Err(UnloweredShader::DescriptorHeap);
    }
    if !shader.root_cbvs.is_empty() || !shader.root_srvs.is_empty() || !shader.root_uavs.is_empty()
        || used.iter().any(|u| u.root == RootType::Descriptor)
    {
        return Err(UnloweredShader::RootView);
    }
    if used.iter().any(|u| u.root == RootType::Table) {
        return Err(UnloweredShader::DescriptorTable);
    }
    if !used.is_empty() {
        return Err(UnloweredShader::UnknownUserData);
    }
    pack_constants(&shader.user_data, shader.user_data_dwords(), block, values)
}

fn pack_constants(
    metadata: &[UserData],
    size: u32,
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
) -> Result<Option<RootWords>, UnloweredShader> {
    if metadata.iter().any(|u| u.root != RootType::Constants) {
        return Err(UnloweredShader::UnknownUserData);
    }
    if metadata.is_empty() && block.is_none() {
        return Ok(None);
    }
    let Some((parameter, count)) = block else {
        return Err(UnloweredShader::ConstantLayout);
    };
    if count != size || !(1..=64).contains(&count) {
        return Err(UnloweredShader::ConstantLayout);
    }
    let mut words = vec![0; count as usize];
    let mut occupied = 0u64;
    for u in metadata {
        let w = words.get_mut(u.index as usize).ok_or(UnloweredShader::ConstantLayout)?;
        let mask = 1u64 << u.index;
        if occupied & mask != 0 {
            return Err(UnloweredShader::ConstantLayout);
        }
        occupied |= mask;
        *w = values(u.slot, u.dword)
            .ok_or(UnloweredShader::MissingConstant { slot: u.slot, dword: u.dword })?;
    }
    Ok(Some(RootWords { parameter, words }))
}

#[cfg(test)]
#[test]
fn closed_draw_constant_admission_never_invents_shader_values() {
    let u = UserData { index: 2, slot: 1, root: RootType::Constants, dword: 3 };
    let result = pack_constants(&[u], 4, Some((5, 4)), |slot, dword| {
        (slot == 1 && dword == 3).then_some(42)
    }).unwrap().unwrap();
    assert_eq!((result.parameter, result.words), (5, vec![0, 0, 42, 0]));
    assert!(matches!(
        pack_constants(&[u], 4, Some((5, 4)), |_, _| None),
        Err(UnloweredShader::MissingConstant { slot: 1, dword: 3 })
    ));
    assert!(matches!(
        pack_constants(&[UserData { root: RootType::Table, ..u }], 4, Some((5, 4)), |_, _| Some(0)),
        Err(UnloweredShader::UnknownUserData)
    ));
    assert!(matches!(
        pack_constants(&[u, u], 4, Some((5, 4)), |_, _| Some(0)),
        Err(UnloweredShader::ConstantLayout)
    ));
}

#[cfg(test)]
#[test]
fn closed_draw_rejects_unlowered_shader_accesses() {
    let shader = HostShader {
        stage: victory_shader::HwStage::Ps,
        dxil: Vec::new(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
        view_marks: false,
    };
    assert!(matches!(constants_only(&shader, &[], None, |_, _| None), Ok(None)));
    for (variant, expected) in [
        (HostShader { memory: true, ..shader.clone() }, UnloweredShader::DynamicAddress),
        (HostShader { memory_writes: true, ..shader.clone() }, UnloweredShader::DynamicAddress),
        (HostShader { view_addresses: true, ..shader.clone() }, UnloweredShader::DynamicAddress),
        (HostShader { heap: true, ..shader.clone() }, UnloweredShader::DescriptorHeap),
        (HostShader { view_marks: true, ..shader.clone() }, UnloweredShader::DescriptorHeap),
        (HostShader { root_uavs: vec![3], ..shader.clone() }, UnloweredShader::RootView),
    ] {
        assert_eq!(constants_only(&variant, &[], None, |_, _| None).err(), Some(expected));
    }
    let used = UsedResource {
        kind: victory_shader::ResourceType::Srv, root: RootType::Table,
        table_offset_dwords: 0, lower_bound: 0, stride: 8, dwords: 8, user_data: 0, slot: 0,
    };
    assert_eq!(
        constants_only(&shader, &[used], None, |_, _| None).err(),
        Some(UnloweredShader::DescriptorTable),
    );
}
