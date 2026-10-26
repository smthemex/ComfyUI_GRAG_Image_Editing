# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""GRAG nodes on the native comfy DiT stack.
"""

from comfy_api.latest import  io
from . import grag_core

# ----------------------------------------------------------------------------------
# nodes
# ----------------------------------------------------------------------------------

class GRAGPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GRAGPatch",
            display_name="GRAG Patch (Group-Relative Attention Guidance)",
            category="GRAG",
            description=(
                "Group-Relative Attention Guidance (arXiv 2510.24657) on a native comfy qwen-image-edit "
                "model. No weights, VAE, CLIP or sampler are replaced: the node only rewrites the joint "
                "attention **keys** after RoPE.\n"
                "\n"
                "k' = cond_b * mean(k_ref) + (k_ref - mean(k_ref)) * cond_delta\n"
                "\n"
                "The paper splits each key into a shared, layer-only bias (the model's intrinsic editing "
                "behaviour) plus a per-token delta (the content-specific edit signal). Re-weighting the "
                "delta modulates how strongly the model follows the edit versus how strongly it holds on "
                "to the reference image.\n"
                "\n"
                "cond_delta < 1 : the per-token deviation shrinks, the reference group's attention "
                "signal flattens and the model is freed from the reference -> stronger edit.\n"
                "cond_delta > 1 : the deviation is amplified -> the output locks onto the reference.\n"
                "Verified on Qwen-Image-Edit-2511 (cfg 4.0, 24 steps): delta=1.70 -> corr 0.995 with the "
                "input (near copy), delta=1.10 -> corr 0.997 (mild), delta=0.90 -> corr 0.928, "
                "delta=0.80 -> corr 0.711 (image drifts far from the reference).\n"
                "Recommended sweep (upstream): 0.8 ~ 1.7 in 0.01 steps, default 1.10."
                "cond_b (upstream 'bias scale') is kept at 1.0 in every released example - leave it alone "
                "unless you are deliberately deviating from the paper.\n"
                "\n"
                "cond_b == 1.0 and cond_delta == 1.0 is a bit-exact no-op, so the node is safe to leave "
                "wired in."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("cond_delta", default=1.10, min=0.0, max=4.0, step=0.01,
                               tooltip="Deviation scale of the reference-image key group. <1 flattens the "
                                       "group signal -> the edit gets stronger (reference holds less); "
                                       ">1 amplifies it -> the output locks onto the reference image. "
                                       "1.0 = disabled. Upstream default 1.10."),
                io.Float.Input("cond_b", default=1.0, min=0.0, max=4.0, step=0.01,
                               tooltip="Bias scale (lambda) of the reference-image group. Upstream keeps this "
                                       "at 1.0 everywhere; changing it rescales the shared edit-behaviour "
                                       "vector instead of the per-token signal."),
                io.Int.Input("img_len", default=4096, min=0, max=65536, step=16,
                             tooltip="How many trailing image tokens form the 'reference group'. Upstream "
                                     "hardcodes 4096 (a 1024x1024 grid). 0 = the entire image block, which is "
                                     "what you want above 1024x1024."),
                io.Float.Input("txt_delta", default=1.0, min=0.0, max=4.0, step=0.01,
                               tooltip="Deviation scale of the text group. Upstream sets it to 1.0 (no-op); "
                                       "<1 dampens the instruction stream, >1 amplifies it."),
                io.Float.Input("txt_b", default=1.0, min=0.0, max=4.0, step=0.01,
                               tooltip="Bias scale of the text group (upstream: 1.0)."),
                io.Float.Input("apply_from", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Start of the sigma window, as denoising progress (0 = pure noise). "
                                       "0.0 = from the first step, matching upstream."),
                io.Float.Input("apply_to", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="End of the sigma window. A 4-step distilled LoRA often looks better "
                                       "with something like 0.0 -> 0.6 because late steps are already "
                                       "committed. 1.0 = to the last step, matching upstream."),
                io.Boolean.Input("verbose", default=False, optional=True),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, cond_delta, cond_b, img_len, txt_delta, txt_b,
                apply_from, apply_to, verbose=False) -> io.NodeOutput:
        state = grag_core.GRAGState(
            lambda_val=cond_b,
            delta_val=cond_delta,
            img_len=img_len,
            txt_lambda=txt_b,
            txt_delta=txt_delta,
            window=(apply_from, apply_to),
        )

        if not state.math_active():
            # Exact identity: hand the MODEL straight back, no clone and no hooks.
            # Cloning a ModelPatcher is not free of side effects on comfy's model
            # management (a clone is tracked separately and can be loaded/streamed
            # differently), and an attn1_patch entry also changes what
            # merge_nested_dicts/cast_transformer_options walk - both showed up as a
            # visible dmax in the A/B even though GRAG itself is a no-op here.
            if verbose:
                print("[GRAG] cond_b=1.0 and cond_delta=1.0 -> pass-through, nothing installed")
            return io.NodeOutput(model)

        m = model.clone()
        grag_core.install_hooks(m, state)
        grag_core.install_step_wrapper(m, state, window=(apply_from, apply_to))

        # stash so GRAGInspect can read the effective config back
        m.model_options.setdefault("grag_state", state)

        if verbose:
            print(f"[GRAG] patched: cond_b={cond_b:.3f} cond_delta={cond_delta:.3f} "
                  f"img_len={img_len} txt=({txt_b:.3f},{txt_delta:.3f}) "
                  f"window={apply_from:.2f}..{apply_to:.2f}")
        return io.NodeOutput(m)

