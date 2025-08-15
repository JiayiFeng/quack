import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math as dialects_math
import torch
from cutlass.cute.runtime import from_dlpack
import math


@cute.jit
def _warp_max(
    val: cute.TensorSSA | cute.Numeric,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = _warp_max(res[i], width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = max(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def _quant_x_to_fp8(
    x: cute.TensorSSA,
    eps: cutlass.Float32,
    num_quant_chunks_per_warp: cutlass.Constexpr[int],
) -> [cute.TensorSSA, cute.TensorSSA]:
    x_abs = cute.TensorSSA(dialects_math.absf(x), x.shape, x.dtype)
    scale = x_abs.reduce(
        cute.ReductionOp.MAX,
        init_val=cutlass.BFloat16(0.0),
        reduction_profile=((None, (0, None)), None, None),
    )
    scale = _warp_max(scale, 32 // num_quant_chunks_per_warp)
    scale = scale.to(cutlass.Float32) / cutlass.Float32(448.0) + eps
    inv_scale = cutlass.Float32(1.0) / scale

    # reshape scale to the x's shape
    inv_scale_t = cute.make_fragment(inv_scale.shape, inv_scale.dtype)
    inv_scale_t.store(inv_scale)
    inv_scale = cute.make_tensor(
        inv_scale_t.iterator, cute.make_layout(x.shape, stride=((0, (0, 1)), 0, 0))
    ).load()

    return (x * inv_scale).to(cute.Float8E4M3FN), scale


@cute.kernel
def _swiglu_and_mul_probs_and_quant_input_kernel(
    x: cute.Tensor,
    probs: cute.Tensor,
    out: cute.Tensor,
    x_fp8_out: cute.Tensor,
    scale_out: cute.Tensor,
    eps: cutlass.Constexpr[float],
    m: cutlass.Constexpr[int],
    num_elems_per_thread: cutlass.Constexpr[int],
    warp_tile_n: cutlass.Constexpr[int],
    block_tiler_mn: cutlass.Constexpr[cute.Shape],
    tv_layout: cute.Layout,
    scale_tiler_mn: cutlass.Constexpr[cute.Shape],
    scale_tv_layout: cute.Layout,
):
    block_idx_x, block_idx_y, _ = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    lane_idx = cute.arch.lane_idx()

    bf16_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type)
    fp32_copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), probs.element_type
    )
    fp8_copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), x_fp8_out.element_type
    )

    x_thread_copy = cute.make_tiled_copy(
        bf16_copy_atom, tv_layout, block_tiler_mn
    ).get_slice(thread_idx)

    prob_thread_copy = cute.make_tiled_copy(
        fp32_copy_atom, tv_layout, block_tiler_mn
    ).get_slice(thread_idx)

    fp8_thread_copy = cute.make_tiled_copy(
        fp8_copy_atom, tv_layout, block_tiler_mn
    ).get_slice(thread_idx)

    scale_thread_copy = cute.make_tiled_copy(
        fp32_copy_atom, scale_tv_layout, scale_tiler_mn
    ).get_slice(thread_idx)

    for block_coord_x in cutlass.range(
        block_idx_x, m // block_tiler_mn[0], cute.arch.grid_dim()[0]
    ):
        g_x0 = cute.local_tile(x, block_tiler_mn, (block_coord_x, block_idx_y))
        g_x1 = cute.local_tile(
            x, block_tiler_mn, (block_coord_x, block_idx_y + cute.arch.grid_dim()[1])
        )
        g_probs = cute.local_tile(probs, block_tiler_mn, (block_coord_x, block_idx_y))
        g_out = cute.local_tile(out, block_tiler_mn, (block_coord_x, block_idx_y))
        g_x0_fp8_out = cute.local_tile(
            x_fp8_out, block_tiler_mn, (block_coord_x, block_idx_y)
        )
        g_x1_fp8_out = cute.local_tile(
            x_fp8_out,
            block_tiler_mn,
            (block_coord_x, block_idx_y + cute.arch.grid_dim()[1]),
        )
        g_scale0_out = cute.local_tile(
            scale_out, scale_tiler_mn, (block_coord_x, block_idx_y, 0)
        )
        g_scale1_out = cute.local_tile(
            scale_out,
            scale_tiler_mn,
            (block_coord_x, block_idx_y + cute.arch.grid_dim()[1], 0),
        )

        t_g_x0, t_g_x1 = [x_thread_copy.partition_S(g_x) for g_x in (g_x0, g_x1)]
        t_g_probs = prob_thread_copy.partition_S(g_probs)
        t_g_out = x_thread_copy.partition_D(g_out)
        t_g_x0_fp8_out = fp8_thread_copy.partition_D(g_x0_fp8_out)
        t_g_x1_fp8_out = fp8_thread_copy.partition_D(g_x1_fp8_out)
        t_g_scale0_out = scale_thread_copy.partition_D(g_scale0_out)
        t_g_scale1_out = scale_thread_copy.partition_D(g_scale1_out)

        r_x0, r_x1, r_out, r_probs, r_x0_fp8, r_x1_fp8 = [
            cute.make_fragment_like(t)
            for t in (
                t_g_x0,
                t_g_x1,
                t_g_out,
                t_g_probs,
                t_g_x0_fp8_out,
                t_g_x1_fp8_out,
            )
        ]

        cute.copy(bf16_copy_atom, t_g_x0, r_x0)
        cute.copy(bf16_copy_atom, t_g_x1, r_x1)
        cute.copy(fp32_copy_atom, t_g_probs, r_probs)

        x0, x1, probs_ = [r.load() for r in (r_x0, r_x1, r_probs)]

        x1_sigmoid = (
            cute.tanh(x1 * cutlass.BFloat16(0.5), fastmath=True).to(cutlass.BFloat16)
            + cutlass.BFloat16(1.0)
        ) * cutlass.BFloat16(0.5)
        out_ = ((x1_sigmoid * x1 * x0).to(cutlass.Float32) * probs_).to(
            cutlass.BFloat16
        )

        r_out.store(out_)
        cute.copy(bf16_copy_atom, r_out, t_g_out)

        x0_fp8, scale0 = _quant_x_to_fp8(x0, eps, warp_tile_n // 128)
        r_x0_fp8.store(x0_fp8)
        cute.copy(fp8_copy_atom, r_x0_fp8, t_g_x0_fp8_out)

        if lane_idx * num_elems_per_thread % 128 == 0:
            r_scale0 = cute.make_fragment_like(t_g_scale0_out)
            r_scale0.store(scale0)
            cute.copy(fp32_copy_atom, r_scale0, t_g_scale0_out)

        x1_fp8, scale1 = _quant_x_to_fp8(x1, eps, warp_tile_n // 128)
        r_x1_fp8.store(x1_fp8)
        cute.copy(fp8_copy_atom, r_x1_fp8, t_g_x1_fp8_out)

        if lane_idx * num_elems_per_thread % 128 == 0:
            r_scale1 = cute.make_fragment_like(t_g_scale1_out)
            r_scale1.store(scale1)
            cute.copy(fp32_copy_atom, r_scale1, t_g_scale1_out)


@cute.jit
def _swiglu_and_mul_probs_and_quant_input_jit_func(
    x: cute.Tensor,
    probs: cute.Tensor,
    out: cute.Tensor,
    x_fp8_out: cute.Tensor,
    scale_out: cute.Tensor,
    m: cutlass.Constexpr[int],
    n: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
    warp_tile_m: cutlass.Constexpr[int],
    warp_tile_n: cutlass.Constexpr[int],
    num_warps_per_block: cutlass.Constexpr[int],
    num_warps_per_sm: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
):
    block_tiler_mn = (warp_tile_m, warp_tile_n * num_warps_per_block)
    num_blocks = num_sms * num_warps_per_sm
    grid_dim_y = n // block_tiler_mn[1]
    grid_dim_x = num_blocks // grid_dim_y
    num_elems_per_thread = warp_tile_n // 32

    thr_layout = cute.make_layout(shape=(1, num_warps_per_block * 32), stride=(32, 1))
    val_layout = cute.make_layout(
        shape=(warp_tile_m, num_elems_per_thread), stride=(num_elems_per_thread, 1)
    )
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    probs_layout = cute.append(probs.layout, cute.make_layout(n, stride=0))
    probs = cute.make_tensor(probs.iterator, probs_layout)

    num_thread_per_quant_chunk = 128 // num_elems_per_thread
    scale_out_layout = cute.append(
        scale_out.layout, cute.make_layout(num_thread_per_quant_chunk, stride=0)
    )
    scale_out = cute.make_tensor(scale_out.iterator, scale_out_layout)

    scale_thr_layout = cute.make_layout(
        shape=(
            1,
            num_warps_per_block * 32 // num_thread_per_quant_chunk,
            num_thread_per_quant_chunk,
        ),
        stride=(1, num_thread_per_quant_chunk, 1),
    )
    scale_val_layout = cute.make_layout(shape=(warp_tile_m, 1, 1), stride=(1, 1, 1))
    scale_tiler_mn, scale_tv_layout = cute.make_layout_tv(
        scale_thr_layout, scale_val_layout
    )

    _swiglu_and_mul_probs_and_quant_input_kernel(
        x,
        probs,
        out,
        x_fp8_out,
        scale_out,
        eps,
        m,
        num_elems_per_thread,
        warp_tile_n,
        block_tiler_mn,
        tv_layout,
        scale_tiler_mn,
        scale_tv_layout,
    ).launch(
        grid=(grid_dim_x, grid_dim_y, 1),
        block=(num_warps_per_block * 32, 1, 1),
    )


def _convert_fp8e4m3fn_tensor(t: torch.Tensor) -> cute.Tensor:
    cute_tensor = from_dlpack(t.view(torch.int8), assumed_align=16)
    cute_tensor.element_type = cute.Float8E4M3FN
    return cute_tensor


_compile_cache = {}


def swiglu_and_mul_probs_and_quant_input(
    x: torch.Tensor,
    probs: torch.Tensor,
    out: torch.Tensor,
    x_fp8_out: torch.Tensor,
    scale_out: torch.Tensor,
    m: int,
    n: int,
    eps: float = 1e-15,
    num_sms: int = 132,
):
    x_ = from_dlpack(x, assumed_align=16)
    probs_ = from_dlpack(probs, assumed_align=16)
    out_ = from_dlpack(out, assumed_align=16)
    x_fp8_out_ = _convert_fp8e4m3fn_tensor(x_fp8_out)
    scale_out_ = from_dlpack(scale_out, assumed_align=16)

    warp_tile_m, warp_tile_n = 2, 256
    assert (
        m % warp_tile_n == 0 and n % warp_tile_m == 0
    ), "m and n must be multiples of warp_tile_n and warp_tile_m respectively"
    num_warps_per_block = min(8, n // warp_tile_n)
    num_warps_per_sm = 64

    compile_key = (m, n, eps)
    if compile_key not in _compile_cache:
        _compile_cache[compile_key] = cute.compile(
            _swiglu_and_mul_probs_and_quant_input_jit_func,
            x_,
            probs_,
            out_,
            x_fp8_out_,
            scale_out_,
            m,
            n,
            eps,
            warp_tile_m,
            warp_tile_n,
            num_warps_per_block,
            num_warps_per_sm,
            num_sms,
        )
    _compile_cache[compile_key](x_, probs_, out_, x_fp8_out_, scale_out_)
