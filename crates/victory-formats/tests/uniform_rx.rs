//! Synthetic block data only; no GPU capture or within-block address oracle.

use victory_formats::texture::{Desc, Dimension, Layout, Swizzle, copy_uniform_rx_rgba8_blocks};

const BLOCK_BYTES: usize = 64 * 1024;
const WIDTH: usize = 257;
const HEIGHT: usize = 129;
const ROW_BYTES: usize = WIDTH * 4;
const SENTINEL: u8 = 0xa5;
const COLORS: [[u8; 4]; 6] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    [16, 17, 18, 19],
    [20, 21, 22, 23],
];

fn desc() -> Desc {
    Desc {
        dimension: Dimension::Tex2d,
        width: WIDTH as u32,
        height: HEIGHT as u32,
        depth: 1,
        mips: 1,
        element_bytes: 4,
        block: 1,
        samples: 1,
        swizzle: Swizzle::SW_64KB_R_X,
    }
}

fn six_blocks() -> Vec<u8> {
    let mut data = vec![0; 6 * BLOCK_BYTES];
    for (block, color) in data.chunks_exact_mut(BLOCK_BYTES).zip(COLORS) {
        for word in block.chunks_exact_mut(4) {
            word.copy_from_slice(&color);
        }
    }
    data
}

fn assert_rejected(layout: &Layout, data: &[u8], dest_len: usize) {
    let mut dest = vec![SENTINEL; dest_len];
    assert!(copy_uniform_rx_rgba8_blocks(layout, data, &mut dest).is_err());
    assert!(
        dest.iter().all(|&b| b == SENTINEL),
        "partial write on rejection"
    );
}

#[test]
fn six_uniform_blocks_become_two_cropped_bands_of_dense_rows() {
    let layout = Layout::new(&desc()).unwrap();
    let n = ROW_BYTES * HEIGHT;
    let mut dest = vec![SENTINEL; n + 7];
    copy_uniform_rx_rgba8_blocks(&layout, &six_blocks(), &mut dest).unwrap();

    // Known 128 x 128 blocks, three across. The right strip is one texel
    // wide, the bottom strip one row tall. Build the oracle as explicit runs,
    // independent of the layout's calculated pitch, offsets and dimensions.
    let row = |left: usize| {
        [
            COLORS[left].repeat(128),
            COLORS[left + 1].repeat(128),
            COLORS[left + 2].to_vec(),
        ]
        .concat()
    };
    let upper = row(0);
    let lower = row(3);
    for (y, actual) in dest[..n].chunks_exact(ROW_BYTES).enumerate() {
        assert_eq!(
            actual,
            if y < 128 { &upper[..] } else { &lower[..] },
            "row {y}"
        );
    }
    assert_eq!(&dest[n..], &[SENTINEL; 7]);
}

#[test]
fn last_block_with_one_visible_texel_still_requires_all_its_words_to_match() {
    let layout = Layout::new(&desc()).unwrap();
    let mut data = six_blocks();
    // Only one of the corner block's 16384 words is visible. Regardless of
    // which word that is, checking visible words alone could never reject it.
    *data.last_mut().unwrap() ^= 1;
    assert_rejected(&layout, &data, ROW_BYTES * HEIGHT + 7);
}

#[test]
fn full_plane_and_full_dense_destination_are_both_required() {
    let layout = Layout::new(&desc()).unwrap();
    let data = six_blocks();
    assert_rejected(&layout, &data[..data.len() - 1], ROW_BYTES * HEIGHT);
    assert_rejected(&layout, &data, ROW_BYTES * HEIGHT - 1);
}

#[test]
fn other_surface_descriptions_are_not_rx_rgba8_single_planes() {
    let d = desc();
    for other in [
        Desc {
            swizzle: Swizzle::SW_64KB_D,
            ..d
        }, // non-XOR, same block size
        Desc {
            swizzle: Swizzle::SW_64KB_D_X,
            ..d
        }, // XOR is not sufficient either
        Desc {
            dimension: Dimension::Tex3d,
            ..d
        },
        Desc { depth: 2, ..d },
        Desc { mips: 2, ..d },
        Desc { block: 4, ..d },
        Desc {
            element_bytes: 8,
            ..d
        },
        Desc { samples: 2, ..d },
    ] {
        let layout = Layout::new(&other).unwrap();
        // Oversize, uniform buffers: neither capacity nor content should be
        // the reason for rejection of an out-of-scope description.
        let n = (layout.size as usize).max(6 * BLOCK_BYTES);
        assert_rejected(&layout, &vec![0x33; n], n);
    }
}

#[test]
fn inconsistent_public_geometry_is_not_trusted() {
    let layout = Layout::new(&desc()).unwrap();
    let data = six_blocks();
    let mut bad_pitch = layout.clone();
    bad_pitch.mips[0].pitch -= 128; // would alias the right block with the next band
    assert_rejected(&bad_pitch, &data, ROW_BYTES * HEIGHT);
    let mut no_mip = layout;
    no_mip.mips.clear();
    assert_rejected(&no_mip, &data, ROW_BYTES * HEIGHT);
}
