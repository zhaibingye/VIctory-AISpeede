# Series S memory

AISpeede has one guest profile: Series S. The hardware has **10 GiB shared
by the title and system**. Keep reporting the existing **7,776 MiB retail
title allowance** to game code; raising that allowance to 10 GiB is not
memory optimization.

The host target on a discrete GPU is **10 GiB extra physical system RAM
and at most 5 GiB of dedicated VRAM**, independently. The first backs
title memory; VRAM is reserved for GPU-native resources and not exposed
as extra memory to the title. Neither is preallocated. The 15 GiB sum
is a secondary check. Windows can migrate allocations and the driver
and other processes are outside the allocator's control.

The GPU resource allocator uses the lower of the current dedicated DXGI
budget and 5 GiB, leaving 384 MiB for costs outside its ledger. It admits
committed textures, sparse texture pages, depth twins and certain
GPU-only temporary buffers on the local segment only while budget remains.
It asks for a non-local GPU-accessible allocation otherwise. Non-local
allocations cost physical RAM and can be slower; no resource is discarded
just to pass the local budget check. The retained charge follows the D3D12
object even when a title object has been released but a descriptor or GPU
list still holds it. This is a managed allocation limit, not a guarantee
about the driver's total dedicated usage. The RAM figure is a target,
not a hard cap on the entire Windows process.

The monitor samples physical system memory against a baseline before
loading the title and queries local DXGI usage and budget. Shared GPU
memory is not counted again. Other processes affect the RAM estimate and
peaks between samples can be missed. No combined estimate is emitted on
an integrated GPU.

The three small idle caches (upload chunks, snapshots, CPU texture
scratch) retain at most 256 MiB in total with plenty of headroom. At
3 GiB available RAM they reduce to a quarter of that, and keep nothing at
2 GiB available RAM or when the observed system/path target is exceeded.
One-shot bindless texture uploads allocate only for changed content, then
release the reordered upload bytes after the GPU fence. A replayable
command list still pins its upload address until reset; the host texture
and the original guest data remain intact in both cases.

## Experimental direct mapping

Set `VICTORY_IN_PLACE=1` to try opening selected guest ranges directly
instead of allocating a shadow for them. The driver must support D3D12
subrange heaps. No whole-allocation fallback: rejected, oversized,
or low-headroom ranges retain the existing copy path. Limit mappings to
16 MiB each and 512 MiB of live mappings, with reservations held until the
last list or view releases the backing. The mapped count measures ranges
shared by CPU and GPU, not additional physical memory.

This is opt-in. In particular, a GPU write made directly into shared pages
cannot be intercepted and withheld like a write made into a shadow, so the
default path remains preferable if rendering regresses. Unset the variable
to return to it. Neither path has been verified to stay under 15 GiB.

## VNME submission counters

A build with the experimental `paged-title` feature can log one line per
retired native submission with `VICTORY_VNME_TIMING=1`. The
`victory_gpu::vnme_timing` target reports CPU command encoding/enqueue/wait
and retirement time, whether the submission used the overlap window,
transferred canonical page bytes, and current RAM/local VRAM ledger use.

CPU wait is not GPU time or frame time. Retirement can include an extra
readback fence for low-level bitmap-tracked diagnostics. Page bytes exclude native texture
transfers and descriptor/scratch traffic. Ledger samples are managed
allocation readings at retirement, not process/driver usage or global peaks.
The option is off by default.
