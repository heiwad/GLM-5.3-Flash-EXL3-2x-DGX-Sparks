#!/usr/bin/env python3
"""Deterministic kpool indexer top-k (VLLM_GLM5NEXT_DETERMINISTIC_TOPK, default 1).

The GLM5Next kpool sparse indexer selects ``select_k = index_topk // kpool``
pools per query row with ``top_k_per_row_prefill`` / ``top_k_per_row_decode``
(or ``persistent_topk``). Those radix/histogram kernels return the right pool
SET but in a launch-dependent ORDER (output slots are claimed with
atomicAdd), and when several pools tie at the k-th score they keep a
launch-dependent SUBSET of them. The pools expand to token indices in list
order and sparse MLA accumulates keys in that order, so identical requests
past ``index_topk`` (2048) tokens round differently at the first sparse
layer and then select different keys from the next layer on (Kindling
measured prompt-logprob drift up to 9.8 nats between identical requests).

This kit additionally keeps only the first ``select_k - 1`` pool ids
(Dockerfile: ``pool_ids[:, : select_k - 1]`` so pools + tail fit the
2048-wide sparse MLA buffer), i.e. it drops whichever pool the kernel happened
to write last.

With the patch (default on) every kpool top-k row is rewritten into one
canonical form: the exact top-``k_out`` pools by score, ties at the k-th score
going to the lowest pool id, ids ascending, ``-1`` padded, where
``k_out = select_k - 1`` for kpool > 1 (the pools the kit actually keeps) and
``topk_tokens`` otherwise.

- prefill: ``top_k_per_row_prefill`` is asked for ``TIE_SLACK`` (32) extra
  candidates (capped at 2048, the kernel's final-stage limit), then a Triton
  pass selects/sorts; a row whose k-th score ties with its worst candidate is
  rescanned in full.
- decode: always ``top_k_per_row_decode`` with the same extra candidates (the
  kit already disables ``persistent_topk`` on GB10 in
  patch_glm_video_placeholders.py; the deterministic branch never takes it),
  then the same Triton pass with each row's effective length computed exactly
  as the kernel does (2-D seq_lens: per row; 1-D: ``seq_len - next_n + j + 1``).

``VLLM_GLM5NEXT_DETERMINISTIC_TOPK=0`` restores the stock dispatch byte for
byte (read once per process).

Provenance: the approach (extra candidates, lowest-id tie-break, ascending
rows) is Kindling's (kindlingai/glm-5.3-flash-gx10 280e72f, unlicensed, so no
code is taken from it). The Triton kernel ``canonicalize_pool_topk`` is from
SpaceshipCreative/glm-5.3-flash-spark patch 0006
(vllm/models/glm5next/nvidia/ops/pool_topk.py, Apache-2.0, Copyright
contributors to the vLLM project / Scott Jordan), inlined here unchanged
except for names, and re-anchored on vLLM 487ecf187's file layout
(model_executor/layers/sparse_attn_indexer_kpool.py).

Same contract as the other kpool patchers: every anchor exactly once (the
decode dispatch line may already carry patch_glm_video_placeholders.py's
``if False and`` edit), idempotent via MARK, compile-checked, atomic replace.
Applied at image build and again by GLM53_OVERLAY_ORDER at container start.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_SPARSE_INDEXER_KPOOL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/"
        "model_executor/layers/sparse_attn_indexer_kpool.py",
    )
)

# Identity string for start.sh validate_overlay_artifacts. Also the boot log tag.
MARK = "# [glm53-det-topk] deterministic kpool top-k (VLLM_GLM5NEXT_DETERMINISTIC_TOPK)\n"

HELPER_ANCHOR = "@eager_break_during_capture\ndef sparse_attn_indexer_kpool(\n"

HELPER = MARK + '''# Ported from SpaceshipCreative/glm-5.3-flash-spark patch 0006
# (vllm/models/glm5next/nvidia/ops/pool_topk.py, Apache-2.0); approach from
# Kindling 280e72f. See overlay/patch_kpool_deterministic_topk.py.
from vllm.triton_utils import tl, triton  # noqa: E402

_GLM53_DET_TOPK = os.environ.get("VLLM_GLM5NEXT_DETERMINISTIC_TOPK", "1") != "0"
# Extra candidates asked of the top-k kernel so ties at the k-th score usually
# resolve without a row rescan; the kernels' final stage holds <= 2048 items.
_GLM53_TOPK_TIE_SLACK = 32
_GLM53_TOPK_MAX_CAND = 2048
_GLM53_SCAN_BLOCK = 1024


def _glm53_det_topk_enabled() -> bool:
    return _GLM53_DET_TOPK


@triton.jit
def _glm53_score_sort_key(x):
    # Reinterpret float32 as int32 so signed integer order matches float order
    # (negative floats have their magnitude bits flipped).
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _glm53_pick(values, offs, i):
    # values[i] of a 1-D block as a scalar.
    return tl.sum(tl.where(offs == i, values, 0), axis=0)


@triton.jit
def _glm53_canonicalize_pool_topk_kernel(
    cand_ptr,
    cand_row_stride,
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    row_start_ptr,
    row_end_ptr,
    out_ptr,
    out_row_stride,
    num_cand,
    k,
    HAS_ROW_START: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
):
    INT32_MAX = 2147483647
    INT32_MIN = -2147483648
    row = tl.program_id(0).to(tl.int64)
    row_end = tl.load(row_end_ptr + row).to(tl.int32)
    if HAS_ROW_START:
        row_start = tl.load(row_start_ptr + row).to(tl.int32)
    else:
        row_start = row_end - row_end
    row_len = row_end - row_start
    score_row = logits_ptr + row * logits_row_stride + row_start * logits_col_stride
    out_row = out_ptr + row * out_row_stride

    offs = tl.arange(0, BLOCK_C)
    ids = tl.load(
        cand_ptr + row * cand_row_stride + offs, mask=offs < num_cand, other=-1
    )
    ok = (ids >= 0) & (ids < row_len)
    n_ok = tl.sum(ok.to(tl.int32), axis=0)
    score = tl.load(score_row + ids * logits_col_stride, mask=ok, other=0.0)
    key = tl.where(ok, _glm53_score_sort_key(score.to(tl.float32)), INT32_MIN)

    # Score key of the k-th best candidate and of the worst valid candidate.
    key_desc = tl.sort(key, descending=True)
    kth_key = _glm53_pick(key_desc, offs, k - 1)
    worst_key = _glm53_pick(key_desc, offs, n_ok - 1)
    above = tl.sum((ok & (key > kth_key)).to(tl.int32), axis=0)
    need = k - above

    # Pools outside the candidates score at most worst_key. When the k-th score
    # equals it and the row has pools beyond the candidates, some tied pools
    # may be missing from the candidates, so rescan the row.
    rescan = (
        (n_ok == num_cand) & (row_len > n_ok) & (n_ok >= k) & (kth_key == worst_key)
    )
    if rescan:
        written = tl.zeros((), dtype=tl.int32)
        ties_left = need
        for off in range(0, row_len, SCAN_BLOCK):
            col = off + tl.arange(0, SCAN_BLOCK)
            in_row = col < row_len
            col_key = _glm53_score_sort_key(
                tl.load(score_row + col * logits_col_stride, mask=in_row, other=0.0).to(
                    tl.float32
                )
            )
            is_tie = in_row & (col_key == kth_key)
            take = (in_row & (col_key > kth_key)) | (
                is_tie & (tl.cumsum(is_tie.to(tl.int32), axis=0) <= ties_left)
            )
            dst = written + tl.cumsum(take.to(tl.int32), axis=0) - 1
            tl.store(out_row + dst, col, mask=take & (dst < k))
            written += tl.sum(take.to(tl.int32), axis=0)
            ties_left -= tl.sum(is_tie.to(tl.int32), axis=0)
        for off in range(0, k, SCAN_BLOCK):
            dst = off + tl.arange(0, SCAN_BLOCK)
            tl.store(
                out_row + dst,
                tl.full((SCAN_BLOCK,), -1, tl.int32),
                mask=(dst >= written) & (dst < k),
            )
    else:
        # Lowest ids win among candidates tied at the k-th score: the need-th
        # smallest tied id is the cut.
        tied_ids = tl.sort(tl.where(ok & (key == kth_key), ids, INT32_MAX))
        tie_cut = _glm53_pick(tied_ids, offs, need - 1)
        keep = ok & (
            (n_ok <= k) | (key > kth_key) | ((key == kth_key) & (ids <= tie_cut))
        )
        kept = tl.sort(tl.where(keep, ids, INT32_MAX))
        tl.store(out_row + offs, tl.where(kept == INT32_MAX, -1, kept), mask=offs < k)


def _glm53_canonicalize_pool_topk(
    candidates: torch.Tensor,
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Write each row's canonical top-k (k = out.shape[1]) into ``out``.

    candidates: [rows, C] int32 ids relative to each row's start, the exact
    top-C of the row in any order, -1 padded, C >= k. logits: [rows, L] fp32,
    row r covers columns [row_starts[r], row_ends[r]) (row_starts None = 0).
    out: [rows, k] int32 (row stride free, unit column stride), must not
    alias candidates; receives the top-k ids ascending, ties at the k-th
    score to the lowest ids, -1 padded.
    """
    rows, num_cand = candidates.shape
    k = out.shape[1]
    assert out.shape[0] == rows and logits.shape[0] >= rows
    assert 0 < k <= num_cand
    assert candidates.dtype == torch.int32 and out.dtype == torch.int32
    assert candidates.stride(1) == 1 and out.stride(1) == 1
    assert row_ends.shape[0] >= rows
    if rows == 0:
        return
    block_c = triton.next_power_of_2(num_cand)
    _glm53_canonicalize_pool_topk_kernel[(rows,)](
        candidates,
        candidates.stride(0),
        logits,
        logits.stride(0),
        logits.stride(1),
        row_ends if row_starts is None else row_starts,
        row_ends,
        out,
        out.stride(0),
        num_cand,
        k,
        HAS_ROW_START=row_starts is not None,
        BLOCK_C=block_c,
        SCAN_BLOCK=_GLM53_SCAN_BLOCK,
        num_warps=8 if block_c >= 2048 else 4,
    )


def _glm53_num_candidates(select_k: int) -> int:
    return max(select_k, min(select_k + _GLM53_TOPK_TIE_SLACK, _GLM53_TOPK_MAX_CAND))


def _glm53_topk_out(topk_dst: torch.Tensor, index_kpool: int) -> torch.Tensor:
    """The columns the kit keeps: the first select_k - 1 pool ids (the
    Dockerfile slices the expanded pool list to that width) for kpool > 1;
    the full row otherwise. The dropped last column is set to -1."""
    if index_kpool > 1:
        topk_dst[:, -1:].fill_(-1)
        return topk_dst[:, : topk_dst.shape[1] - 1]
    return topk_dst


def _glm53_decode_row_ends(
    seq_lens: torch.Tensor, next_n: int, num_rows: int
) -> torch.Tensor:
    """Per-row effective length, exactly as topKPerRowDecode computes it."""
    if seq_lens.dim() == 2:
        ends = seq_lens.reshape(-1)[:num_rows]
    else:
        idx = torch.arange(num_rows, device=seq_lens.device)
        ends = seq_lens[idx // next_n] - next_n + (idx % next_n) + 1
    return ends.clamp(min=0).to(torch.int32).contiguous()


'''

PREFILL_OLD = """            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
"""
PREFILL_NEW = """            elif _glm53_det_topk_enabled():
                # [glm53-det-topk] exact top-k_out, lowest pool id wins ties,
                # ids ascending (see the helper block above).
                _glm53_ncand = _glm53_num_candidates(select_k)
                _glm53_cand = torch.full(
                    (num_rows, _glm53_ncand), -1, dtype=torch.int32, device=logits.device
                )
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    _glm53_cand,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    _glm53_ncand,
                )
                _glm53_canonicalize_pool_topk(
                    _glm53_cand,
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    _glm53_topk_out(topk_dst, index_kpool),
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
"""

DECODE_PREFIX = """        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

"""
DISPATCH_STOCK = "        if current_platform.is_cuda() and select_k in (512, 1024, 2048):\n"
# patch_glm_video_placeholders.py (runtime) rewrites the stock line to this.
DISPATCH_VIDEO = (
    "        if False and current_platform.is_cuda() and "
    "select_k in (512, 1024, 2048):  # GB10 persistent_topk smem\n"
)
DECODE_DET = """        if _glm53_det_topk_enabled() and current_platform.is_cuda():
            # [glm53-det-topk] top_k_per_row_decode with extra candidates
            # (never persistent_topk), then the canonical top-k_out rewrite.
            _glm53_ncand = _glm53_num_candidates(select_k)
            _glm53_cand = torch.full(
                (num_rows, _glm53_ncand), -1, dtype=torch.int32, device=logits.device
            )
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                _glm53_cand,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                _glm53_ncand,
            )
            _glm53_canonicalize_pool_topk(
                _glm53_cand,
                logits,
                None,
                _glm53_decode_row_ends(seq_lens, next_n, num_rows),
                _glm53_topk_out(topk_dst, index_kpool),
            )
        el"""

# The kit's own Dockerfile edit this patch depends on (k_out = select_k - 1).
KIT_SLICE = "pool_ids[:, : select_k - 1]"


def verified_state(text: str) -> bool:
    return (
        text.count(MARK) == 1
        and text.count("def _glm53_canonicalize_pool_topk(") == 1
        and text.count("            elif _glm53_det_topk_enabled():\n") == 1
        and text.count("        if _glm53_det_topk_enabled() and current_platform.is_cuda():\n") == 1
        and (
            text.count("        elif current_platform.is_cuda() and select_k in (512, 1024, 2048):\n")
            + text.count(
                "        elif False and current_platform.is_cuda() and "
                "select_k in (512, 1024, 2048):  # GB10 persistent_topk smem\n"
            )
        ) == 1
        and text.count(KIT_SLICE) == 2
    )


def prepare(source: str) -> tuple[str, str]:
    marker_count = source.count(MARK)
    if marker_count:
        if marker_count != 1 or not verified_state(source):
            raise ValueError(
                f"partial/inconsistent deterministic top-k patch (marker={marker_count})"
            )
        return source, "already present"
    if "deterministic_topk" in source or "canonicalize_pool_topk" in source:
        raise ValueError(
            "this vLLM already carries a deterministic kpool top-k "
            "(upstream/SSC 0006 layout?); refusing to stack a second one"
        )
    counts = {
        "helper": source.count(HELPER_ANCHOR),
        "prefill": source.count(PREFILL_OLD),
        "decode_prefix": source.count(DECODE_PREFIX),
        "kit_slice": source.count(KIT_SLICE),
    }
    dispatch_stock = source.count(DECODE_PREFIX + DISPATCH_STOCK)
    dispatch_video = source.count(DECODE_PREFIX + DISPATCH_VIDEO)
    if (
        counts["helper"] != 1
        or counts["prefill"] != 1
        or counts["decode_prefix"] != 1
        or counts["kit_slice"] != 2
        or dispatch_stock + dispatch_video != 1
    ):
        raise ValueError(
            "sparse_attn_indexer_kpool anchors drifted (expected vLLM 487ecf187 "
            f"plus the kit Dockerfile select_k - 1 slices): {counts} "
            f"dispatch_stock={dispatch_stock} dispatch_video={dispatch_video}"
        )
    patched = source.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    patched = patched.replace(PREFILL_OLD, PREFILL_NEW, 1)
    dispatch = DISPATCH_STOCK if dispatch_stock else DISPATCH_VIDEO
    # "        if <dispatch>" -> "<det branch>        elif <dispatch>"
    patched = patched.replace(
        DECODE_PREFIX + dispatch,
        DECODE_PREFIX + DECODE_DET + dispatch.lstrip(" "),
        1,
    )
    if not verified_state(patched):
        raise ValueError("deterministic top-k post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-det-topk.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob("sparse_attn_indexer_kpool*.pyc"):
        pyc.unlink(missing_ok=True)


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"deterministic kpool top-k preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    mode = os.environ.get("VLLM_GLM5NEXT_DETERMINISTIC_TOPK", "1")
    print(
        f"[glm53-det-topk] {TARGET.name}: {action} "
        f"(VLLM_GLM5NEXT_DETERMINISTIC_TOPK={mode!r}; '0' = stock dispatch)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
