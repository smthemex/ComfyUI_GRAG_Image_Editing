# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""GRAG (Group-Relative Attention Guidance) — native comfy DiT core implementation.

Reference implementation: https://github.com/little-misfit/GRAG-Image-Editing
Paper: arXiv 2510.24657 "Group Relative Attention Guidance for Image Editing"

------------------------------------------------------------------------------
1. What upstream actually does
------------------------------------------------------------------------------
``hacked_models/models.py::QwenDoubleStreamAttnProcessor2_0.__call__``::

    img_query/key/value = to_q/k/v(hidden_states)        # image stream
    txt_query/key/value = add_q/k/v(encoder_hidden)      # text stream
    ... norm_q / norm_k / norm_added_q / norm_added_k ...
    img_q,k = apply_rotary_emb_qwen(..., img_freqs)      # <-- RoPE here
    txt_q,k = apply_rotary_emb_qwen(..., txt_freqs)

    if grag_scale != (0, 0):
        txt_len, txt_bias, txt_delta = txt_srag_scale    # ((512, 1.0, 1.0))
        img_len, img_bias, img_delta = cond_srag_scale   # ((4096, cond_b, cond_delta))
        txt_key_mean  = txt_key[:, :txt_len, :, :].mean(dim=1)     # [B, H, D]
        cond_key_mean = img_key[:, -img_len:, :, :].mean(dim=1)
        txt_key[:, :txt_len, :, :] = txt_bias * txt_key_mean + (txt_key - txt_key_mean) * txt_delta
        img_key[:, -img_len:, :, :] = img_bias  * cond_key_mean + (img_key - cond_key_mean) * img_delta

    joint_q = cat([txt_q, img_q], dim=1)   # text first, image after
    ...

i.e. GRAG rewrites the **post-RoPE** joint key as

    k' = bias * mean(k_group) + (k - mean(k_group)) * delta

The paper tunes the *image* group (``cond_b`` / ``cond_delta``); the text group is
left at ``(1.0, 1.0)`` which is an exact identity.

------------------------------------------------------------------------------
2. Where it hooks into comfy without touching a single core file
------------------------------------------------------------------------------
``comfy/ldm/modules/attention.py::wrap_attn`` calls
``transformer_options["optimized_attention_override"](func, q, k, v, heads, **kw)``
before dispatch (the same route ``comfy_extras/nodes_sparse_attention.py`` uses).
At that point q/k/v are already QK-normed **and RoPE'd**, in ``[B, H, S, D]`` with
``skip_reshape=True`` — which is exactly the tensor upstream reweights, so the port
is bit-exact with no RoPE replication needed.

Measured on this machine (``_grag_override_probe5.py``, random weights so the
rotation is observable)::

    dmax(captured_k , pre-rope k) = 8.46875
    dmax(captured_k , post-rope k) = 0.0
    => override sees POST-rope keys

The text/image split is not in ``transformer_options`` at that point (comfy writes
``img_slice`` into a *local* copy for ``attn1_patch`` only), so a tiny
``attn1_patch`` recorder caches ``seq_txt`` from the immediately preceding
``attn1_patch`` call of the same attention block. Recorder and override are strictly
sequential inside ``Attention.forward``, and a tick/consume pair makes a stale value
impossible: if the override runs without a fresh tick it degrades to a no-op.
"""

import torch


# ----------------------------------------------------------------------------------
# core math
# ----------------------------------------------------------------------------------

def grag_reweight(key, txt_len, img_len, lambda_val, delta_val,
                  txt_lambda=1.0, txt_delta=1.0):
    """Group-relative reweighting of a joint attention key ``[B, H, S, D]``.

    ``txt_len``  : number of leading text tokens (sequence is ``[text | image]``).
    ``img_len``  : how many *trailing* image tokens to reweight, 0 = the whole group.
                   Upstream hardcodes 4096; at 1024² a qwen edit grid is exactly 4096
                   tokens so the default covers everything, and the clamp below keeps
                   larger grids from silently reaching into the text part.
    ``lambda_val`` / ``delta_val`` : image-group bias / deviation scale (cond_b, cond_delta).
    ``txt_lambda`` / ``txt_delta`` : text-group pair, upstream leaves it at (1, 1).

    All of it is pure elementwise arithmetic on already-materialised q/k/v, so it
    costs two means and two FMA passes per block per step.
    """
    txt = key[:, :, :txt_len, :]
    img = key[:, :, txt_len:, :]

    if img_len and img_len > 0 and img_len < img.shape[2]:
        img_head = img[:, :, :img.shape[2] - img_len, :]
        img_tail = img[:, :, img.shape[2] - img_len:, :]
    else:
        img_head = None
        img_tail = img

    if lambda_val != 1.0 or delta_val != 1.0:
        img_mean = img_tail.mean(dim=2, keepdim=True)
        img_tail = lambda_val * img_mean + (img_tail - img_mean) * delta_val

    if txt_lambda != 1.0 or txt_delta != 1.0:
        txt_mean = txt.mean(dim=2, keepdim=True)
        txt = txt_lambda * txt_mean + (txt - txt_mean) * txt_delta

    if img_head is None:
        return torch.cat([txt, img_tail], dim=2)
    return torch.cat([txt, img_head, img_tail], dim=2)


# ----------------------------------------------------------------------------------
# shared state
# ----------------------------------------------------------------------------------

class GRAGState:
    """One instance per patched MODEL; mutated per step, read by the hooks."""

    def __init__(self, lambda_val=1.0, delta_val=1.10, img_len=4096,
                 txt_lambda=1.0, txt_delta=1.0, window=(0.0, 1.0)):
        self.lambda_val = float(lambda_val)
        self.delta_val = float(delta_val)
        self.img_len = int(img_len)
        self.txt_lambda = float(txt_lambda)
        self.txt_delta = float(txt_delta)
        # normalised progress window (0 = pure noise, 1 = clean)
        self.window = (min(window), max(window))

        self.enabled = True          # toggled by the step wrapper (sigma window)
        self.step = 0
        self.sigma = None
        self.txt_len = None          # written by the attn1 recorder
        self._tick = 0               # recorder increments
        self._consumed = 0           # override reads + consumes

        self.blocks_fired = 0

    # -- recorder side ---------------------------------------------------------
    def note_txt_len(self, txt_len):
        self.txt_len = int(txt_len)
        self._tick += 1

    def fresh_txt_len(self):
        """Return a txt_len that arrived after the last override call, else None."""
        if self._tick == self._consumed or self.txt_len is None:
            return None
        self._consumed = self._tick
        return self.txt_len

    # -- gating ----------------------------------------------------------------
    def math_active(self):
        """True when the user asked for a non-identity transform."""
        return (self.lambda_val != 1.0 or self.delta_val != 1.0
                or self.txt_lambda != 1.0 or self.txt_delta != 1.0)

    def active(self):
        return self.enabled and self.math_active()


# ----------------------------------------------------------------------------------
# hooks
# ----------------------------------------------------------------------------------

def make_attn1_recorder(state):
    """``attn1_patch`` that only caches ``seq_txt`` (does not touch q/k/v).

    comfy calls it as ``patch(q, k, v, pe=..., attn_mask=..., extra_options=...)``
    with ``extra_options["img_slice"] = [seq_txt, total]``, immediately before the
    RoPE + ``optimized_attention`` of the very same block.
    """

    def recorder(q, k, v, pe=None, attn_mask=None, extra_options=None):
        eo = extra_options or {}
        img_slice = eo.get("img_slice")
        if img_slice is not None:
            state.note_txt_len(int(img_slice[0]))
        return {}

    recorder.is_grag_recorder = True
    return recorder


def make_attention_override(state, previous=None):
    """``optimized_attention_override`` that applies GRAG then delegates to comfy.

    ``func`` is the selected attention kernel's inner function, so plain delegation
    keeps whatever backend the launch flags picked. An override installed earlier in
    the chain is chained through, so other MODEL patches keep working.
    """

    def override(func, *args, **kwargs):
        passthrough = (previous, func, args, kwargs) if previous is not None else (None, func, args, kwargs)

        if not state.active() or len(args) < 4:
            return _delegate(passthrough)

        q, k, v = args[0], args[1], args[2]
        if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
            return _delegate(passthrough)

        txt_len = state.fresh_txt_len()
        if txt_len is None or txt_len <= 0 or txt_len >= k.shape[2]:
            return _delegate(passthrough)

        new_k = grag_reweight(
            k, txt_len, state.img_len,
            state.lambda_val, state.delta_val,
            state.txt_lambda, state.txt_delta,
        )
        state.blocks_fired += 1
        return _delegate((previous, func, (q, new_k, v) + tuple(args[3:]), kwargs))

    override.is_grag = True
    return override


def _delegate(payload):
    previous, func, args, kwargs = payload
    if previous is not None:
        return previous(func, *args, **kwargs)
    return func(*args, **kwargs)


def install_hooks(patcher, state):
    """Attach recorder + override to a cloned ModelPatcher (idempotent)."""
    to = patcher.model_options.setdefault("transformer_options", {})

    patches = to.setdefault("patches", {})
    existing = patches.get("attn1_patch") or []
    if not any(getattr(p, "is_grag_recorder", False) for p in existing):
        patches["attn1_patch"] = list(existing) + [make_attn1_recorder(state)]

    if state.math_active():
        current = to.get("optimized_attention_override")
        if not getattr(current, "is_grag", False):
            to["optimized_attention_override"] = make_attention_override(state, previous=current)

    return patcher


# ----------------------------------------------------------------------------------
# per-step gate
# ----------------------------------------------------------------------------------

def make_step_wrapper(state, total_steps=None, start_percent=0.0, end_percent=1.0, previous=None):
    """Model-function wrapper that reports progress and gates GRAG by sigma.

    ``params`` is the dict comfy builds in ``samplers.py``:
    ``{"input", "timestep", "c", "cond_or_uncond"}``; ``timestep`` is the current
    sigma (flow matching, ``ModelSamplingFlux.timestep(σ) == σ``).

    The gate exists because a 4-step distilled LoRA spends most of its budget below
    σ≈0.3, where the edit is already committed and reweighting only adds noise. The
    default window is the full run, which reproduces upstream exactly.
    """
    lo, hi = state.window

    def wrapper(model_function, params):
        sigma = params.get("timestep")
        if torch.is_tensor(sigma) and sigma.numel() == 1:
            s = float(sigma.reshape(-1)[0].item())
            state.sigma = s
            if total_steps and total_steps > 1:
                state.step = int(round((1.0 - s) * (total_steps - 1)))
            progress = 1.0 - s
            state.enabled = (lo <= progress <= hi) if (lo > 0.0 or hi < 1.0) else True
        if previous is not None:
            return previous(model_function, params)
        return model_function(params["input"], params["timestep"], **params["c"])

    return wrapper


def install_step_wrapper(patcher, state, total_steps=None, window=(0.0, 1.0)):
    state.window = (min(window), max(window))
    if state.window == (0.0, 1.0):
        # nothing to gate: leave any pre-existing wrapper untouched
        return
    previous = patcher.model_options.get("model_function_wrapper")
    patcher.set_model_unet_function_wrapper(
        make_step_wrapper(state, total_steps=total_steps,
                          start_percent=state.window[0], end_percent=state.window[1],
                          previous=previous)
    )
