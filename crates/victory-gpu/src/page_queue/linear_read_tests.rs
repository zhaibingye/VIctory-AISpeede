use super::*;
use crate::descriptor::View;
use crate::object::{IUNKNOWN, S_OK, This};
use crate::page_draw::{ColorClear, ColorTarget};
use crate::page_resource::ResourcePlan;
use crate::page_slabs::PageSlabs;
use crate::page_texture::TextureInput;
use crate::resource;
use victory_memory::{
    PAGE_BYTES,
    physical::{BackingPins, PageNumbers},
    virtual_pages::VirtualPages,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

// 走真实按地址创建及 SRV 来源选择。VA 只用于布局和合成页表，不读取进程内存。
struct AddressTexture(This);

impl AddressTexture {
    fn new(va: u64, flags: D3D12_RESOURCE_FLAGS) -> Self {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 2,
            Height: 2,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut object = std::ptr::null();
        // ResourceDesc 与 D3D12_RESOURCE_DESC 共享 C ABI；创建时保留来宾的显式线性布局。
        assert_eq!(
            resource::create_at_address(
                std::ptr::null(),
                va,
                (&raw const desc).cast(),
                0,
                std::ptr::null(),
                &IUNKNOWN,
                &mut object
            ),
            S_OK
        );
        Self(object)
    }

    fn linear_srv(&self) -> TextureInput {
        let srv = View::Shader(self.0 as usize, None)
            .paged_texture_srv(false)
            .unwrap();
        let TextureInput::Linear { copy, .. } = &srv else {
            panic!("应选用来宾线性字节");
        };
        assert!(copy.uses_unproven_alias_backing());
        srv
    }
}

impl Drop for AddressTexture {
    fn drop(&mut self) {
        resource::resource_release(self.0);
    }
}

fn recording(host: &Host, batch: &PinnedBatch) -> HostList {
    let list = host.new_list().unwrap();
    let (_, _, first) = batch.pages().next().unwrap();
    assert!(
        batch
            .pages()
            .all(|(_, _, page)| first.slab() == page.slab())
    );
    // 安全：局部页池只有一个新 slab，在录制上传和清除期间由 batch 保留。
    unsafe {
        list.recording().ResourceBarrier(&[transition(
            first.buffer(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
    }
    list
}

#[test]
fn external_linear_srv_checks_the_alias_writer_before_native_enqueue() {
    const VA: u64 = 0x3510 * PAGE_BYTES;
    const ALIAS: u64 = VA + PAGE_BYTES;
    let host = crate::host::host().expect("D3D12 宿主");
    let source = AddressTexture::new(VA, D3D12_RESOURCE_FLAG_NONE);
    let writer = AddressTexture::new(ALIAS, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    let srv = source.linear_srv();
    let copy = resource::host_copy(source.0).unwrap();
    let target = resource::host_copy(writer.0).unwrap();
    assert_ne!(srv.host_texture(), &target.texture);
    assert_eq!(target.candidate_guest_span.unwrap().0, ALIAS);
    let clear = ColorClear {
        target: ColorTarget::capture_candidate(target, Default::default()).unwrap(),
        color: [1., 0., 0., 1.],
        rects: Vec::new(),
    };

    let id = PageNumbers::default().allocate(1).unwrap();
    let (mut backing, _section) =
        crate::title_backing::test_capture(id, &[VA as usize, ALIAS as usize]);
    let mut ranges = Vec::new();
    srv.ranges(&mut ranges).unwrap();
    assert_eq!(ranges, [(VA, VA + 8), (VA + 256, VA + 264)]);
    let heap = crate::memory::heap().unwrap();
    let mut queue = host.direct_queue();
    // 全套测试可能共用宿主：不重置之前真实提交过的写者标记。
    let prior = host
        .unclassified_native_writes
        .load(std::sync::atomic::Ordering::Relaxed);
    for with_writer in [false, true] {
        let result = host.execute_paged_work_locked(
            &mut queue,
            heap,
            &mut backing,
            &ranges,
            WriteSet::Declared,
            |list, _, batch, pins, _, plan| {
                assert_eq!(
                    Some(batch.pinned_at(VA).unwrap().physical()),
                    pins.expected(ALIAS as usize)
                );
                if with_writer {
                    clear.encode(host, list, batch, plan)?;
                }
                assert!(!plan.authoritative(&copy)); // 写者不是读取视图的宿主对象。
                let (uploaded, _) = srv.encode(host, list, batch, plan, None)?;
                assert_ne!(&uploaded, srv.host_texture());
                Ok(BTreeSet::new())
            },
        );
        if with_writer || prior {
            assert!(
                matches!(result, Err(BatchError::UnclassifiedSharedTextureWriter {
                current_submission, previous_submission,
            }) if current_submission == with_writer && previous_submission == prior),
                "{result:?}"
            );
        } else {
            assert_eq!(result.unwrap().downloaded, 0);
        }
    }
}

#[test]
fn external_linear_srv_rejects_a_second_row_without_its_own_reservation() {
    const BOUNDARY: u64 = 0x3520 * PAGE_BYTES;
    let host = crate::host::host().expect("D3D12 宿主");
    let source = AddressTexture::new(BOUNDARY - 256, D3D12_RESOURCE_FLAG_NONE);
    let srv = source.linear_srv();
    let mut ranges = Vec::new();
    srv.ranges(&mut ranges).unwrap();
    assert_eq!(
        ranges,
        [(BOUNDARY - 256, BOUNDARY - 248), (BOUNDARY, BOUNDARY + 8)]
    );
    let mut vm = VirtualPages::default();
    let mut pins = BackingPins::default();
    for start in [BOUNDARY - PAGE_BYTES, BOUNDARY] {
        let va = start as usize;
        if start < BOUNDARY {
            vm.reserved(va);
        }
        vm.committed(va, va, PAGE_BYTES as usize);
        pins.remember_with_origin(
            va,
            va + PAGE_BYTES as usize,
            Some(vm.page(va).unwrap()),
            vm.has_own_reservation(va),
        );
    }
    let mut slabs = PageSlabs::default();
    let batch = PinnedBatch::from_capture(host, &mut slabs, &pins, &ranges).unwrap();
    assert!(batch.verify_independent_texture_backing(ranges[0]).is_ok());
    assert!(batch.verify_independent_texture_backing(ranges[1]).is_err());
    let list = recording(host, &batch);
    let mut plan = ResourcePlan::default();
    assert!(
        matches!(
            srv.encode(host, &list, &batch, &mut plan, None),
            Err(BatchError::UnsupportedTextureMap { .. })
        ),
        "第二行虽已固定，但其私有页无独立保留区"
    );
    assert!(!plan.section_texture_read);
    drop(list);
    batch.cancel(&mut slabs);
}
