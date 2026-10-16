#!/usr/bin/env python3
"""Checks for overlay/patch_kpool_deterministic_topk.py.

CPU (always): the patch applies to vLLM 487ecf187's
sparse_attn_indexer_kpool.py after the kit's Dockerfile ``select_k - 1``
edits, is idempotent, refuses anchor drift, and commutes with
patch_glm_video_placeholders.py's persistent_topk edit (either order gives
the same file).

Kernel (only when torch + triton can run it: a CUDA device, or
TRITON_INTERPRET=1 on CPU): the canonicalizer returns the exact top-k with
ties to the lowest id, ascending, -1 padded, for ragged prefill rows and
dense decode rows, and its output does not depend on the candidate order or
on WHICH tied pools the top-k kernel happened to keep (the rescan path).

GLM53_SPARSE_INDEXER_KPOOL_PY=<file> also preflights a real installed file.
Run: python3 tests/test_kpool_deterministic_topk.py
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH_DIR = next(
    p for p in (HERE, ROOT / "overlay") if (p / "patch_kpool_deterministic_topk.py").is_file()
)
sys.path.insert(0, str(PATCH_DIR))
import patch_kpool_deterministic_topk as P  # noqa: E402

FIXTURE = next(
    p
    for p in (
        HERE / "fixtures" / "sparse_attn_indexer_kpool-487ecf187.py.txt",
        HERE / "sparse_attn_indexer_kpool-487ecf187.py.txt",
    )
    if p.is_file()
)
VIDEO_OLD = "if current_platform.is_cuda() and select_k in (512, 1024, 2048):"
VIDEO_NEW = (
    "if False and current_platform.is_cuda() and "
    "select_k in (512, 1024, 2048):  # GB10 persistent_topk smem"
)


def kit_dockerfile_state(text: str) -> str:
    """The two Dockerfile edits the kit makes to this file at build time."""
    for old, new in (
        (
            "                    expanded = expand_pools_and_append_tail(\n"
            "                        pool_ids, q_seq, index_kpool\n"
            "                    )\n",
            "                    expanded = expand_pools_and_append_tail(\n"
            "                        pool_ids[:, : select_k - 1], q_seq, index_kpool\n"
            "                    )\n",
        ),
        (
            "            out = expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)\n",
            "            out = expand_pools_and_append_tail(\n"
            "                pool_ids[:, : select_k - 1], dec_seq, index_kpool\n"
            "            )\n",
        ),
    ):
        assert text.count(old) == 1
        text = text.replace(old, new)
    return text


def video(text: str) -> str:
    assert VIDEO_OLD in text
    return text.replace(VIDEO_OLD, VIDEO_NEW, 1)


def check_cpu() -> None:
    stock = kit_dockerfile_state(FIXTURE.read_text())
    patched, action = P.prepare(stock)
    assert action == "patched"
    compile(patched, "sparse_attn_indexer_kpool.py", "exec")
    assert P.prepare(patched) == (patched, "already present")
    # Stock dispatch untouched when the flag is off: the else branches keep
    # the original calls byte for byte.
    assert patched.count("torch.ops._C.persistent_topk(") == 1
    stock_calls = "                    topk_dst,\n                    num_rows,"
    assert patched.count(stock_calls) == stock.count(stock_calls) == 4
    # Commutes with the runtime persistent_topk disable.
    a = video(patched)
    b, action_b = P.prepare(video(stock))
    assert action_b == "patched"
    assert a == b, "video patch and det top-k patch do not commute"
    assert P.verified_state(a)
    compile(a, "sparse_attn_indexer_kpool.py", "exec")
    assert P.prepare(a) == (a, "already present")
    # Drift: without the kit's select_k - 1 slices k_out would be wrong.
    raw = FIXTURE.read_text()
    try:
        P.prepare(raw)
    except ValueError:
        pass
    else:
        raise AssertionError("must refuse a file without the kit's select_k - 1 slices")
    try:
        P.prepare(stock.replace("torch.ops._C.top_k_per_row_prefill(", "torch.ops._C.topk_prefill(", 1))
    except ValueError:
        pass
    else:
        raise AssertionError("must refuse prefill anchor drift")
    try:
        P.prepare(patched.replace(P.MARK, P.MARK + P.MARK))
    except ValueError:
        pass
    else:
        raise AssertionError("must refuse a doubled marker")
    real = os.environ.get("GLM53_SPARSE_INDEXER_KPOOL_PY")
    if real:
        text = Path(real).read_text()
        out, act = P.prepare(text)
        compile(out, real, "exec")
        print(f"real target {real}: {act}")
    print("kpool deterministic top-k patch: CPU anchor checks OK")


def _load_helpers():
    import torch
    import triton
    import triton.language as tl

    vtu = types.ModuleType("vllm.triton_utils")
    vtu.tl, vtu.triton = tl, triton
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.triton_utils"] = vtu
    import importlib.util
    import tempfile

    # Triton's JIT needs real source lines, so exec the helper from a file.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "_glm53_det_topk_helpers.py"
        path.write_text("import os\nimport torch\n" + P.HELPER)
        spec = importlib.util.spec_from_file_location("_glm53_det_topk_helpers", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return torch, mod


def _reference(scores, k):
    """Exact top-k ids (ties to lowest id), ascending, -1 padded."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    keep = sorted(order[:k])
    return keep + [-1] * (k - len(keep))


def check_kernel() -> None:
    try:
        torch, H = _load_helpers()
    except Exception as exc:  # noqa: BLE001
        print(f"kernel checks skipped (torch/triton unavailable: {exc})")
        return
    cuda = torch.cuda.is_available()
    if not cuda and os.environ.get("TRITON_INTERPRET") != "1":
        print("kernel checks skipped (no CUDA; set TRITON_INTERPRET=1 to run on CPU)")
        return
    dev = "cuda" if cuda else "cpu"
    g = torch.Generator().manual_seed(0)
    rows, width = 12, 3000
    # Heavy ties: scores drawn from a handful of levels.
    levels = torch.randn(7, generator=g)
    logits = levels[torch.randint(0, 7, (rows, width), generator=g)].float()
    starts = torch.randint(0, 400, (rows,), generator=g).int()
    lens = torch.tensor([0, 1, 50, 300, 510, 511, 512, 600, 1500, 2000, 2400, 2599]).int()
    ends = (starts + lens).int()
    for k, slack in ((511, 32), (511, 1), (63, 0)):
        ncand = k + slack
        ref = []
        cands = []
        for r in range(rows):
            s = logits[r, starts[r]:ends[r]].tolist()
            ref.append(_reference(s, k))
            # A kernel-like candidate list: the exact top-ncand SET but with a
            # random subset of the boundary ties and a random order.
            perm = torch.randperm(len(s), generator=g).tolist()
            order = sorted(perm, key=lambda i: -s[i])[:ncand]
            order = [order[i] for i in torch.randperm(len(order), generator=g).tolist()]
            cands.append(order + [-1] * (ncand - len(order)))
        cand = torch.tensor(cands, dtype=torch.int32, device=dev)
        out = torch.full((rows, k + 1), 7, dtype=torch.int32, device=dev)
        H._glm53_canonicalize_pool_topk(
            cand, logits.to(dev), starts.to(dev), ends.to(dev), out[:, :k]
        )
        got = out[:, :k].cpu().tolist()
        for r in range(rows):
            assert got[r] == ref[r], (k, slack, r, got[r][:8], ref[r][:8])
        assert (out[:, k] == 7).all(), "wrote past k"
    # Decode layout: no row starts, rows from _glm53_decode_row_ends.
    seq2d = torch.tensor([[900, 901], [30, 31]], dtype=torch.int32)
    assert H._glm53_decode_row_ends(seq2d, 2, 4).tolist() == [900, 901, 30, 31]
    seq1d = torch.tensor([900, 31], dtype=torch.int32)
    assert H._glm53_decode_row_ends(seq1d, 2, 4).tolist() == [899, 900, 30, 31]
    dl = levels[torch.randint(0, 7, (4, 1000), generator=g)].float()
    ends_d = H._glm53_decode_row_ends(seq2d, 2, 4)
    k = 255
    cands = []
    for r in range(4):
        s = dl[r, : int(ends_d[r])].tolist()
        order = sorted(range(len(s)), key=lambda i: -s[i])[:k + 32]
        cands.append(order[::-1] + [-1] * (k + 32 - len(order)))
    out = torch.empty((4, k), dtype=torch.int32, device=dev)
    H._glm53_canonicalize_pool_topk(
        torch.tensor(cands, dtype=torch.int32, device=dev), dl.to(dev), None, ends_d.to(dev), out
    )
    for r in range(4):
        assert out[r].cpu().tolist() == _reference(dl[r, : int(ends_d[r])].tolist(), k), r
    assert H._glm53_num_candidates(512) == 544 and H._glm53_num_candidates(2048) == 2048
    print(f"kpool deterministic top-k kernel checks OK on {dev}")


if __name__ == "__main__":
    check_cpu()
    check_kernel()
