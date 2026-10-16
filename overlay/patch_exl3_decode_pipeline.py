#!/usr/bin/env python3
"""Add an opt-in SM121 K4/N256 thin-decode MoE pipeline; preserve stock kernels.

Derived from the validated patch in
coolbho3k/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark
(`vendor/miaai-exl3/patch_exl3_decode_pipeline.py`), originally written for
exllamav3 c5d9c65 (v0.0.43) and ported here to the v1.5.0 kernel layout
(0740edc) pinned by this variant's Dockerfile. The v1.5 kernel template is
`<t_bits, MOE_TILESIZE_N, cb, M_TILE>` with a `moe_gemm_tile` helper, a
runtime `gated` flag and the deterministic slot store (output_scratch /
fused_base); the copy keeps all of that, so the fast kernel honours
EXL3_MOE_FUSED_DET exactly like the stock one. NOT re-measured on SM121 after
the port: it stays opt-in (GLM53_EXL3_MOE_FAST=0 default).

What it does (additive only; the stock `exl3_moe` kernels are untouched):
  * copies `quant/exl3_moe_kernel.cuh` to
    `quant/glm53_exl3_moe_fast_kernel.cuh` with a `bool shared_input`
    template parameter, the `glm53_exl3_moe_fast_kernel` entry point and the
    tile helper renamed `glm53_moe_gemm_tile` (its body reads the stage
    macros, so it must not share a name with the stock helper);
  * when `shared_input` is true (and the activation is gated), the redundant
    up input Hadamard is skipped and the up GEMM reads the already-transformed
    gate buffer;
  * adds `quant/comp_units/glm53_exl3_moe_fast.cu` instantiating
    `<4, 256, cb=1 (mcg), M_TILE=16, true/false>` with MOE_FRAG_STAGES 1 /
    MOE_SH_STAGES 8 (stock is 3/3);
  * host dispatch in `quant/exl3_moe.cu` selects the fast kernel only for the
    16-row launch when K == 4, the codebook is mcg (cb_idx 0), N_off == 1
    (hidden % 256 == intermediate % 256 == 0), `GLM53_EXL3_MOE_FAST=1`, and the
    device is SM121 (12.1);
  * `shared_input` is true only when the gate/up SUH pointer tables are the
    identical allocation (aliased in `overlay/exl3.py` after a load-time
    all-expert equality check).

Usage (in the image build, after the aarch64 stub patch):
    python3 patch_exl3_decode_pipeline.py /tmp/exllamav3/exllamav3/exllamav3_ext
"""

from pathlib import Path
import sys


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one native source anchor: {old!r}')
    return text.replace(old, new, 1)


def patch(root):
    root = Path(root)
    quant = root / 'quant'
    host_path = quant / 'exl3_moe.cu'
    host = host_path.read_text()
    if 'GLM53_EXL3_MOE_FAST' in host:
        raise RuntimeError('Decode pipeline patch already present; use a clean source tree')
    kernel = (quant / 'exl3_moe_kernel.cuh').read_text()
    # v1.5 tile helper: rename every use (definition + 6 call sites) so the
    # copy's helper, which expands MOE_SH_STAGES/MOE_FRAG_STAGES at parse time,
    # never shares a name with the stock one.
    n_tile = kernel.count('moe_gemm_tile')
    if n_tile != 7:
        raise RuntimeError(f'Expected 7 moe_gemm_tile references, found {n_tile}')
    kernel = kernel.replace('moe_gemm_tile', 'glm53_moe_gemm_tile')
    kernel = once(kernel, 'template<int t_bits, int MOE_TILESIZE_N, int cb, int M_TILE = MOE_TILESIZE_M>',
                  'template<int t_bits, int MOE_TILESIZE_N, int cb, int M_TILE, bool shared_input>')
    kernel = once(kernel, 'void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)',
                  'void glm53_exl3_moe_fast_kernel(EXL3_MOE_KERNEL_ARGS)')
    redundant = '''                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );'''
    kernel = once(kernel, redundant,
                  '                if (!(shared_input && gated)) {\n' + redundant + '\n                }')
    kernel = once(kernel,
                  'gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);',
                  'gemm_up((shared_input && gated) ? temp_state_g : temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);')
    wrapper = '''#include "exl3_moe_instances.cuh"
#undef MOE_FRAG_STAGES
#define MOE_FRAG_STAGES 1
#undef MOE_SH_STAGES
#define MOE_SH_STAGES 8
#include "../glm53_exl3_moe_fast_kernel.cuh"
// cb = 1 is the mcg codebook instance (exl3_moe.cu: cb_idx 0 -> *_cb1).
fp_exl3_moe_kernel glm53_exl3_moe_fast_k4_n256(bool shared_input) {
    return shared_input ? glm53_exl3_moe_fast_kernel<4, 256, 1, 16, true>
                        : glm53_exl3_moe_fast_kernel<4, 256, 1, 16, false>;
}
'''
    helper = '''
#include <cstdlib>
#include <cstring>

fp_exl3_moe_kernel glm53_exl3_moe_fast_k4_n256(bool shared_input);

static bool glm53_fast_moe_enabled(int device) {
    static const bool requested = [] {
        const char* value = std::getenv("GLM53_EXL3_MOE_FAST");
        TORCH_CHECK(!value || !std::strcmp(value, "0") || !std::strcmp(value, "1"),
                    "GLM53_EXL3_MOE_FAST must be 0 or 1");
        return value && !std::strcmp(value, "1");
    }();
    if (!requested) return false;
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "Unexpected device index");
    static thread_local int supported[MAX_DEVICES] = {};
    if (!supported[device]) {
        int major = 0, minor = 0;
        cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        cuda_check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        supported[device] = major == 12 && minor == 1 ? 1 : -1;
    }
    return supported[device] == 1;
}
'''
    host = once(host, '#include <set>', '#include <set>\n' + helper)
    anchor = '        kernel = exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];'
    host = once(host, anchor, anchor + '''
        if (K == 4 && cb_idx == 0 && N_off == 1 && glm53_fast_moe_enabled(device)) {
            kernel = glm53_exl3_moe_fast_k4_n256(
                gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr());
        }''')
    bindings_path = root / 'bindings.cpp'
    bindings = once(bindings_path.read_text(),
                    '    m.def("exl3_moe", &exl3_moe, "exl3_moe");',
                    '    m.def("exl3_moe", &exl3_moe, "exl3_moe");\n'
                    '    m.def("glm53_fast_moe_version", []() { return 1; });')
    # Validate every anchor before writing any source file.
    (quant / 'glm53_exl3_moe_fast_kernel.cuh').write_text(kernel)
    (quant / 'comp_units/glm53_exl3_moe_fast.cu').write_text(wrapper)
    host_path.write_text(host)
    bindings_path.write_text(bindings)
    print(f'Installed opt-in SM121 K4/N256 decode pipeline into {root}')


if __name__ == '__main__':
    patch(sys.argv[1])
