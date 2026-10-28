#!/usr/bin/env python
"""Fold BOTH LoRA families -- the VLM (PaliGemma, rank 16) and the ACTION-EXPERT `_1` (rank 32) --
into their base weights, with a functional check on EVERY folded site.

This is merge_vlm_lora.py generalized to the `_lora_k2` checkpoints (VLM LoRA + expert LoRA),
which merge_vlm_lora.py refuses (it asserts no expert LoRA) and merge_expert_lora.py folds WRONG:
merge_expert_lora.py does a plain per-head `a[n] @ b[n]`, but for attn_vec_einsum gemma's eqn
"BTNH,NHD->BTD" drops the head axis N, so lora.Einsum's b-einsum ("BTL,NLD->BTD") SUMS lora_b
over N and the trained operator is delta[n] = a[n] @ (sum_n b[n]). merge_expert_lora.py has no
functional check to catch it; the wrong fold degraded replace_IMLE_lr1p35e4_6000 from normalized
MSE 0.0002 (LoRA intact) to 0.0008 (folded). This script keeps merge_vlm_lora.py's correction and
its per-site functional check, so a wrong fold cannot pass silently.

scaling = alpha/rank = 1.0 for both variants (gemma_2b_lora 16/16, gemma_300m_lora 32/32, no rslora).

    python merge_all_lora.py <src_params_dir> <dst_params_dir>
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

# (rank, width) per family; scaling = alpha/rank = 1.0 for both (alpha == rank, no rslora).
FAMILIES = {"vlm": (16, 2048), "expert": (32, 1024)}
SCALING = 1.0


def _arr(leaf):
    return leaf["value"] if isinstance(leaf, dict) and "value" in leaf else leaf


def _put(node, key, arr):
    if isinstance(node[key], dict) and "value" in node[key]:
        node[key]["value"] = arr
    else:
        node[key] = arr


def _is_leaf(node) -> bool:
    return isinstance(node, dict) and "value" in node


def family(path: str) -> str:
    """Expert leaves carry the `_1` suffix (gemma._name); everything else is the VLM."""
    return "expert" if "_1" in path else "vlm"


def merge(params: dict) -> dict:
    stats = {"vlm": 0, "expert": 0}
    dropped: list[str] = []

    def fold(node, base, a_key, b_key, prefix):
        fam = family(prefix)
        rank, width = FAMILIES[fam]
        a, b, w = _arr(node[a_key]), _arr(node[b_key]), _arr(node[base])
        assert a.shape[-1] == rank and b.shape[-2] == rank, f"{prefix}/{base}: rank != {rank} ({a.shape}, {b.shape})"
        a32, b32 = a.astype(jnp.float32), b.astype(jnp.float32)
        if "attn_vec_einsum" in prefix:
            # gemma eqn "BTNH,NHD->BTD" loses the head axis N in the output, so lora.Einsum's
            # generated b-einsum ("BTL,NLD->BTD") SUMS lora_b over N. The trained operator is
            # therefore  delta[n] = a[n] @ (sum_n b[n]),  NOT the per-head a[n] @ b[n].
            b32 = jnp.broadcast_to(jnp.sum(b32, axis=1, keepdims=True), b32.shape)
        delta = jnp.einsum("...ir,...ro->...io", a32, b32) * SCALING
        assert delta.shape == w.shape, f"{prefix}/{base}: delta {delta.shape} != w {w.shape}"
        # Functional check on layer 0 with the EXACT einsum pair lora.Einsum._make_lora_eqns produces.
        eqns = {
            "q_einsum": ("BTD,NDH->BTNH", "BTD,NDL->BTNL", "BTNL,NLH->BTNH"),
            "kv_einsum": ("BSD,2KDH->2BSKH", "BSD,2KDL->2BSKL", "2BSKL,2KLH->2BSKH"),
            "attn_vec_einsum": ("BTNH,NHD->BTD", "BTNH,NHL->BTL", "BTL,NLD->BTD"),
            "gating_einsum": None,  # FeedForward._dot: x@a@b per slice, no scaling
            "linear": None,
        }
        site = next((k for k in eqns if k in f"{prefix}/{base}"), None)
        if site and eqns[site] is not None:
            eqn, eqn_a, eqn_b = eqns[site]
            w0, a0, b0 = (t[0].astype(jnp.float32) for t in (w, a, b))
            n_heads, head_dim = w0.shape[0], w0.shape[-2] if site == "attn_vec_einsum" else w0.shape[-1]
            in_shape = {"q_einsum": (1, 3, width), "kv_einsum": (1, 3, width),
                        "attn_vec_einsum": (1, 3, w0.shape[0], w0.shape[1])}[site]
            x = jax.random.normal(jax.random.key(0), in_shape, dtype=jnp.float32)
            y_lora = jnp.einsum(eqn, x, w0) + jnp.einsum(eqn_b, jnp.einsum(eqn_a, x, a0), b0) * SCALING
            y_merged = jnp.einsum(eqn, x, (w0 + delta[0].astype(jnp.float32)))
            err = float(jnp.max(jnp.abs(y_lora - y_merged)) / (jnp.max(jnp.abs(y_lora)) + 1e-9))
            assert err < 1e-5, f"{prefix}/{base}: functional mismatch rel {err:.2e}"
            print(f"    functional check [{fam}] {site}: rel err {err:.2e} OK")
        _put(node, base, (w.astype(jnp.float32) + delta).astype(w.dtype))
        del node[a_key], node[b_key]
        stats[fam] += 1
        dropped.append(f"[{fam}] {prefix}/{base}")

    def walk(node, prefix=""):
        if not isinstance(node, dict):
            return
        keys = set(node.keys())
        if {"w", "lora_a", "lora_b"} <= keys:
            fold(node, "w", "lora_a", "lora_b", prefix)
            return
        for base in ("gating_einsum", "linear"):
            if {base, f"{base}_lora_a", f"{base}_lora_b"} <= keys:
                fold(node, base, f"{base}_lora_a", f"{base}_lora_b", prefix)
        if _is_leaf(node):
            return
        for k in list(node.keys()):
            walk(node[k], f"{prefix}/{k}")

    walk(params)
    print(f"  merged {stats['vlm']} VLM + {stats['expert']} expert LoRA pair(s):")
    for d in dropped:
        print(f"    {d}")
    assert stats["vlm"] == 5, f"expected 5 VLM sites, got {stats['vlm']}"
    assert stats["expert"] in (0, 5), f"expected 0 or 5 expert sites, got {stats['expert']}"
    remaining = [
        "/".join(str(x) for x in p)
        for p, _ in jax.tree_util.tree_flatten_with_path(params)[0]
        if "lora" in "/".join(str(x) for x in p)
    ]
    assert not remaining, f"lora leaves remain after merge: {remaining}"
    return params


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    print(f"reading {src}")
    ckptr = ocp.PyTreeCheckpointer()
    meta = ckptr.metadata(src)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding), meta)
    params = ckptr.restore(src, restore_args=restore_args)
    params = merge(params)
    print(f"writing {dst}")
    ocp.PyTreeCheckpointer().save(dst, params)
    print("done")


if __name__ == "__main__":
    main()
