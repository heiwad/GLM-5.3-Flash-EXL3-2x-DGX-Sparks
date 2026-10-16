#!/usr/bin/env python3
"""Make ExLlamaV3's extension compile on aarch64 (GB10 / DGX Spark).

The pinned exllamav3 (v1.5.0, 0740edc) ships x86-only CPU code: AVX2/AVX-512
targets, the AVX CPU MoE offload kernels (cpu/moe_mul1.cpp), the native-TP CPU
all-reduce, and two `__builtin_ia32_pause()` spin hints. This kit never uses any
of them (vLLM owns TP; experts stay on the GPU), but they must compile.

This applies upstream's own aarch64 guard, commit 181eea1 ("aarch64 (DGX
Spark/GB10): guard x86 AVX intrinsics, stub CPU MoE offload + native TP CPU
reduce, arm yield spin hint"), which first shipped in v1.5.2 and applies cleanly
to v1.5.0 (no cpu/ or parallel/ commits in between). A tree that already
carries it (v1.5.2+) is left untouched.

The pre-1.5 stub approach (overwriting avx2_target.* with stubs) no longer
works: v1.5 adds is_f16c_supported(), a 6-argument perform_cpu_reduce() and
cpu_reduce_parallel(), so the old stubs fail to compile or leave undefined
symbols that only surface at `import exllamav3_ext`.

Every anchor must match exactly once; nothing is written unless all match.
"""

from pathlib import Path
import sys

MARK = "EXL3_AARCH64_STUB"
X86 = "(defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))"

AVX2_CPU_OLD_1 = """    if (avx2_check) return avx2_supported;
    #ifdef __linux__
        avx2_supported = __builtin_cpu_supports("avx2");
"""
AVX2_CPU_NEW_1 = f"""    if (avx2_check) return avx2_supported;
    #if defined(__linux__) && !{X86}  // {MARK}
        (void)0;  // non-x86 Linux: no AVX
    #elif defined(__linux__)
        avx2_supported = __builtin_cpu_supports("avx2");
"""
AVX2_CPU_OLD_2 = """    if (f16c_check) return f16c_supported;
    #ifdef __linux__
        f16c_supported = __builtin_cpu_supports("f16c");
"""
AVX2_CPU_NEW_2 = f"""    if (f16c_check) return f16c_supported;
    #if defined(__linux__) && !{X86}  // {MARK}
        (void)0;  // non-x86 Linux: no AVX
    #elif defined(__linux__)
        f16c_supported = __builtin_cpu_supports("f16c");
"""
AVX2_H_OLD = """bool is_f16c_supported();

#ifdef __linux__
"""
AVX2_H_NEW = f"""bool is_f16c_supported();

#if defined(__linux__) && {X86}  // {MARK}
"""
AVX512_CPU_OLD = """#ifdef __linux__
    // Check for AVX-512F and AVX-512BW support
"""
AVX512_CPU_NEW = f"""#if defined(__linux__) && !{X86}  // {MARK}
        (void)0;  // non-x86 Linux: no AVX
    #elif defined(__linux__)
    // Check for AVX-512F and AVX-512BW support
"""
AVX512_H_OLD = """bool is_avx512_supported();

#ifdef __linux__
"""
AVX512_H_NEW = f"""bool is_avx512_supported();

#if defined(__linux__) && {X86}  // {MARK}
"""
HANDOFF_OLD = """inline void cpu_pause_()
{
#ifdef __linux__
    __builtin_ia32_pause();
"""
HANDOFF_NEW = """inline void cpu_pause_()
{
#if defined(__aarch64__) || defined(__arm__)  // EXL3_AARCH64_PAUSE
    asm volatile("yield" ::: "memory");
#elif defined(__linux__)
    __builtin_ia32_pause();
"""
REDUCE_OLD = """            if (++spins < 65536)
            {
                #ifdef __linux__
                    __builtin_ia32_pause();
"""
REDUCE_NEW = """            if (++spins < 65536)
            {
                #if defined(__aarch64__) || defined(__arm__)  // EXL3_AARCH64_PAUSE
                    asm volatile("yield" ::: "memory");
                #elif defined(__linux__)
                    __builtin_ia32_pause();
"""

MUL1_HEAD = '#include "moe_mul1.h"\n#include <c10/util/Half.h>\n'
MUL1_TAIL = f"""
#else  // !x86: {MARK} — CPU MoE offload kernels are AVX-only; stubbed on this architecture
#include "moe_mul1.h"
#include <torch/extension.h>
#define NO_MOE_CPU() TORCH_CHECK(false, "CPU MoE offload (moe_mul1) is not available on this CPU architecture (x86 AVX2/AVX-512 only)")
void exl3_moe_cpu_set_prof(bool) {{}}
int64_t exl3_moe_cpu_pool_stress(int, int, int, int) {{ NO_MOE_CPU(); return 0; }}
void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int) {{ NO_MOE_CPU(); }}
bool exl3_moe_cpu_has_avx2() {{ return false; }}
bool exl3_moe_cpu_has_avx512_bw() {{ return false; }}
bool exl3_moe_cpu_has_avx512_vnni() {{ return false; }}
bool exl3_moe_cpu_has_avx512_vbmi() {{ return false; }}
int64_t exl3_moe_cpu_make_layer(
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    int64_t, double, int64_t) {{ NO_MOE_CPU(); return 0; }}
void exl3_moe_cpu_free_layer(int64_t) {{}}
void exl3_moe_cpu_forward_raw(int64_t, const at::Half*, const int32_t*, const at::Half*, float*, int, int, int) {{ NO_MOE_CPU(); }}
void exl3_moe_cpu_forward(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t) {{ NO_MOE_CPU(); }}
#endif
"""
AVX2_REDUCE_HEAD = '#include <immintrin.h>\n#include "all_reduce_cpu_avx2.h"\n'
AVX2_REDUCE_TAIL = f"""
#else  // !x86: {MARK} — CPU all-reduce for native TP backend is AVX-only; stubbed
#include "all_reduce_cpu_avx2.h"
#include <torch/extension.h>
void enable_fast_fp() {{}}
void enable_fast_fp_avx2() {{}}
void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t)
{{ TORCH_CHECK(false, "Native tensor-parallel CPU reduce is not available on this CPU architecture (x86 AVX2 only); use the nccl backend"); }}
void perform_cpu_reduce_avx2(PGContext* c, size_t a, uint32_t b, uint32_t d, uint8_t* e, size_t f) {{ perform_cpu_reduce(c, a, b, d, e, f); }}
void cpu_reduce_parallel(void (*)(uint16_t*, const uint16_t*, const uint16_t*, size_t), void (*)(uint16_t*, const uint16_t*, size_t),
                         uint16_t*, const uint16_t*, const uint16_t*, size_t, int) {{ TORCH_CHECK(false, "cpu_reduce_parallel unavailable on this architecture"); }}
#endif
"""
AVX512_REDUCE_HEAD = '#include <immintrin.h>\n#include "all_reduce_cpu_avx512.h"\n'
AVX512_REDUCE_TAIL = f"\n#endif  // {MARK}\n"


def once(text: str, old: str, new: str, name: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{name}: expected exactly one aarch64 anchor, found {n}: {old!r}")
    return text.replace(old, new, 1)


def wrap(text: str, head: str, tail: str, name: str) -> str:
    if not text.startswith(head):
        raise RuntimeError(f"{name}: unexpected file head (wanted {head!r})")
    if not text.endswith("}\n"):
        raise RuntimeError(f"{name}: unexpected file tail")
    return f"#if {X86}\n" + text + tail


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/exllamav3/exllamav3/exllamav3_ext")
    files = {
        "avx2_target.cpp": root / "avx2_target.cpp",
        "avx2_target.h": root / "avx2_target.h",
        "avx512_target.cpp": root / "avx512_target.cpp",
        "avx512_target.h": root / "avx512_target.h",
        "cpu/moe_handoff.cu": root / "cpu/moe_handoff.cu",
        "cpu/moe_mul1.cpp": root / "cpu/moe_mul1.cpp",
        "parallel/all_reduce_cpu.cu": root / "parallel/all_reduce_cpu.cu",
        "parallel/all_reduce_cpu_avx2.cpp": root / "parallel/all_reduce_cpu_avx2.cpp",
        "parallel/all_reduce_cpu_avx512.cpp": root / "parallel/all_reduce_cpu_avx512.cpp",
    }
    missing = [k for k, p in files.items() if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"not an exllamav3 >= 1.5 extension tree ({root}); missing {missing}. "
            "This patcher targets the v1.5 layout only."
        )
    src = {k: p.read_text() for k, p in files.items()}
    guarded = [k for k, t in src.items() if MARK in t or "EXL3_AARCH64_PAUSE" in t]
    if guarded:
        # Upstream (v1.5.2+) carries 181eea1, later reworded in a few .cpp
        # files; the header guards and the two TUs that fail hardest are the
        # signature. Anything less is a half-patched tree.
        required = {"avx2_target.h", "avx512_target.h", "cpu/moe_mul1.cpp", "cpu/moe_handoff.cu",
                    "parallel/all_reduce_cpu.cu", "parallel/all_reduce_cpu_avx2.cpp"}
        if not required.issubset(guarded):
            raise RuntimeError(f"partially guarded tree (have {guarded}); use a clean source tree")
        print(f"aarch64 guards already present upstream in {root} (v1.5.2+); nothing to do")
        return 0

    out = dict(src)
    out["avx2_target.cpp"] = once(out["avx2_target.cpp"], AVX2_CPU_OLD_1, AVX2_CPU_NEW_1, "avx2_target.cpp")
    out["avx2_target.cpp"] = once(out["avx2_target.cpp"], AVX2_CPU_OLD_2, AVX2_CPU_NEW_2, "avx2_target.cpp")
    out["avx2_target.h"] = once(out["avx2_target.h"], AVX2_H_OLD, AVX2_H_NEW, "avx2_target.h")
    out["avx512_target.cpp"] = once(out["avx512_target.cpp"], AVX512_CPU_OLD, AVX512_CPU_NEW, "avx512_target.cpp")
    out["avx512_target.h"] = once(out["avx512_target.h"], AVX512_H_OLD, AVX512_H_NEW, "avx512_target.h")
    out["cpu/moe_handoff.cu"] = once(out["cpu/moe_handoff.cu"], HANDOFF_OLD, HANDOFF_NEW, "cpu/moe_handoff.cu")
    out["parallel/all_reduce_cpu.cu"] = once(
        out["parallel/all_reduce_cpu.cu"], REDUCE_OLD, REDUCE_NEW, "parallel/all_reduce_cpu.cu"
    )
    out["cpu/moe_mul1.cpp"] = wrap(out["cpu/moe_mul1.cpp"], MUL1_HEAD, MUL1_TAIL, "cpu/moe_mul1.cpp")
    out["parallel/all_reduce_cpu_avx2.cpp"] = wrap(
        out["parallel/all_reduce_cpu_avx2.cpp"], AVX2_REDUCE_HEAD, AVX2_REDUCE_TAIL,
        "parallel/all_reduce_cpu_avx2.cpp",
    )
    out["parallel/all_reduce_cpu_avx512.cpp"] = wrap(
        out["parallel/all_reduce_cpu_avx512.cpp"], AVX512_REDUCE_HEAD, AVX512_REDUCE_TAIL,
        "parallel/all_reduce_cpu_avx512.cpp",
    )
    for k, p in files.items():
        if out[k] != src[k]:
            p.write_text(out[k])
    print(f"aarch64 guards (upstream 181eea1) applied in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
