use super::*;
use crate::page_draw::{ColorClear, ColorTarget};
use crate::page_slabs::PageSlabs;
use crate::page_texture::SharedTextureBytes;
use crate::resource::HostTexture;
use std::sync::Arc;
use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
use victory_memory::physical::{BackingPins, PageNumbers};

const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

fn texture(host: &Host, dimension: D3D12_RESOURCE_DIMENSION, depth: u16,
    flags: D3D12_RESOURCE_FLAGS,
) -> HostTexture {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: dimension, Width: 4, Height: 1, DepthOrArraySize: depth, MipLevels: 1,
        Format: FORMAT, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: flags,
        ..Default::default()
    };
    let mut result = None::<ID3D12Resource>;
    // 安全：创建与测试视图的尺寸和用途匹配的独立宿主副本。
    unsafe { host.device.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut result,
    ) }.unwrap();
    HostTexture::new(result.unwrap(), D3D12_RESOURCE_STATE_COMMON.0 as u32, Vec::new())
}

fn volume(host: &Host, base: u64) -> ShaderTextureUav {
    let layout = Layout::new(&Desc {
        dimension: Dimension::Tex3d, width: 4, height: 1, depth: 2, mips: 1,
        element_bytes: 4, block: 1, samples: 1, swizzle: Swizzle::LINEAR,
    }).unwrap();
    let bytes = SharedTextureBytes::new(base, FORMAT, &layout).unwrap();
    let target = texture(host, D3D12_RESOURCE_DIMENSION_TEXTURE3D, 2,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    // 只合成捕获结果，不注册来宾对象；prepare 仍走真实格式检查、预填充和状态转换。
    ShaderTextureUav(NativeTextureUav::capture(Arc::new(target), FORMAT, None, true).unwrap(), Some(bytes))
}

fn recording(host: &Host, batch: &PinnedBatch) -> HostList {
    let list = host.new_list().unwrap();
    // 测试使用局部页池，至多两页，都在同一个从 COMMON 开始的 slab。
    let (_, _, first) = batch.pages().next().unwrap();
    assert!(batch.pages().all(|(_, _, page)| first.slab() == page.slab()));
    let begin = crate::host::transition(first.buffer(), D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // 安全：batch 的固定页保留这个 slab 直到本命令列表被丢弃。
    unsafe { list.recording().ResourceBarrier(&[begin]); }
    list
}

#[test]
fn shared_volume_prefill_declares_section_read_even_with_a_different_native_writer() {
    const VA: u64 = PAGE_BYTES;
    const ALIAS: u64 = VA + PAGE_BYTES;
    let host = crate::host::host().expect("D3D12 宿主");
    let image = volume(host, VA);
    let ranges: Vec<_> = image.byte_backing().unwrap().ranges().collect();
    let id = PageNumbers::default().allocate(1).unwrap();
    let (mut backing, _section) = crate::title_backing::test_capture(id, &[VA as usize, ALIAS as usize]);
    let mut slabs = PageSlabs::default();
    let batch = PinnedBatch::capture(host, &mut slabs, &mut backing, &ranges).unwrap();
    assert_eq!(batch.pinned_at(VA).unwrap().physical(), backing.pins().expected(ALIAS as usize).unwrap());

    let mut writer = texture(host, D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    writer.candidate_guest_span = Some((ALIAS, ALIAS + 16));
    let clear = ColorClear {
        target: ColorTarget::capture_candidate(Arc::new(writer), Default::default()).unwrap(),
        color: [1., 0., 0., 1.], rects: Vec::new(),
    };
    for with_writer in [true, false] {
        let list = recording(host, &batch);
        let mut plan = ResourcePlan::default();
        if with_writer {
            let _queue = host.direct_queue();
            clear.encode(host, &list, &batch, &mut plan).unwrap();
        }
        assert!(!plan.authoritative(image.target()));
        assert!(!plan.section_texture_read);
        image.prepare(host, &list, &batch, &mut plan, None).unwrap();
        // 两位都取自真实编码，是 submit_work 门禁的输入；不提交到全局 Host，
        // 避免基线漏拒时发布粘性的历史写入标记，也不依赖其他测试的提交顺序。
        assert_eq!((plan.section_texture_read, plan.unclassified_native_write), (true, with_writer));
    }
    batch.cancel(&mut slabs);
}

#[test]
fn shared_volume_prefill_rejects_an_unproven_later_slice_before_transition() {
    let host = crate::host::host().expect("D3D12 宿主");
    let boundary = 2 * PAGE_BYTES;
    let image = volume(host, boundary - 256);
    let ranges: Vec<_> = image.byte_backing().unwrap().ranges().collect();
    assert_eq!(ranges, [(boundary - 256, boundary - 240), (boundary, boundary + 16)]);

    let mut vm = victory_memory::virtual_pages::VirtualPages::default();
    let mut pins = BackingPins::default();
    for (va, independent) in [(PAGE_BYTES as usize, true), (boundary as usize, false)] {
        if independent { vm.reserved(va); }
        vm.committed(va, va, PAGE_BYTES as usize);
        pins.remember_with_origin(va, va + PAGE_BYTES as usize, Some(vm.page(va).unwrap()), vm.has_own_reservation(va));
    }
    let mut slabs = PageSlabs::default();
    let batch = PinnedBatch::from_capture(host, &mut slabs, &pins, &ranges).unwrap();
    assert!(batch.verify_independent_texture_backing(ranges[0]).is_ok());
    assert!(batch.verify_independent_texture_backing(ranges[1]).is_err());
    let list = recording(host, &batch);
    let mut plan = ResourcePlan::default();
    let result = image.prepare(host, &list, &batch, &mut plan, None);
    assert!(matches!(result, Err(BatchError::UnsupportedTextureMap { .. })),
        "两张页都已固定，但第二个 W 分片的私有页没有独立保留区证明");
    assert!(!plan.section_texture_read);
    assert_eq!(plan.state(image.target()), D3D12_RESOURCE_STATE_COMMON);
    drop(list);
    batch.cancel(&mut slabs);
}
