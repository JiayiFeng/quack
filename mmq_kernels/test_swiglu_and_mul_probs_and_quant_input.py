from swiglu_and_mul_probs_and_quant_input import swiglu_and_mul_probs_and_quant_input
from parameterized import parameterized
import torch
from cutlass.cute.runtime import from_dlpack

from mmq_kernels.jit_kernels.swiglu_mul_probs import (
    swiglu_and_mul_probs_and_input_quant as ref_impl,
)
import unittest
import triton


def _n_bytes(a: torch.Tensor):
    return a.element_size() * a.numel()


def _tbps(n_bytes: int, ms: float) -> float:
    return n_bytes * 1e-9 / ms


class TestCuteDSLKernel(unittest.TestCase):
    @parameterized.expand(
        [(m, n) for m in [4096 * x for x in [6, 8]] for n in [512 * x for x in [3, 4]]]
    )
    def test_cute_dsl_kernel(self, m: int, n: int):
        x = torch.randn((m, n * 2), device="cuda", dtype=torch.bfloat16)
        probs = torch.randn((m,), device="cuda", dtype=torch.float32)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        x_fp8_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale_out = torch.empty((m, n * 2 // 128), device="cuda", dtype=torch.float32)

        num_total_bytes = (
            _n_bytes(x)
            + _n_bytes(out)
            + _n_bytes(probs)
            + _n_bytes(x_fp8_out)
            + _n_bytes(scale_out)
        )

        cute_dsl_time = triton.testing.do_bench(
            lambda: swiglu_and_mul_probs_and_quant_input(
                x=x,
                probs=probs,
                out=out,
                x_fp8_out=x_fp8_out,
                scale_out=scale_out,
                m=m,
                n=n,
            )
        )
        print(f"m={m}, n={n} CuTeDSL Kernel:")
        print(f"  Average execution time: {cute_dsl_time:.4f} ms")
        print(f"  Throughput: {_tbps(num_total_bytes, cute_dsl_time):.2f} TB/s")

        out_ref = torch.empty_like(out)
        x_fp8_out_ref = torch.empty_like(x_fp8_out)
        scale_out_ref = torch.empty_like(scale_out)
        ref_time = triton.testing.do_bench(
            lambda: ref_impl(
                x=x,
                probs=probs,
                out=out_ref,
                x_fp8_out=x_fp8_out_ref,
                scale_out=scale_out_ref,
            )
        )
        print(f"m={m}, n={n} Ref Kernel:")
        print(f"  Average execution time: {ref_time:.4f} ms")
        print(f"  Throughput: {_tbps(num_total_bytes, ref_time):.2f} TB/s")

        torch.testing.assert_close(out, out_ref, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(scale_out, scale_out_ref)
        # torch.testing.assert_close(
        #     x_fp8_out.float(), x_fp8_out_ref.float()
        # )
