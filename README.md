# ComfyUI_GRAG_Image_Editing

[GRAG-Image-Editing](https://github.com/little-misfit/GRAG-Image-Editing) : Group-Relative Attention Guidance for Image Editing, on the **native comfy DiT stack**.


# Update
* 改为comfyUI核心封装/change to comfyUI core Wrapper

# Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_GRAG_Image_Editing
```

# Models
----
 
* comfyUI normal：dit, qwen-image vae and qwen_2.5_vl_7b   
* lora, lightx2v 8step lora# 千问edit加速  
```

├── ComfyUI/models/diffusion_models # or gguf
|     ├── qwen_image_edit_2511_fp8mixed.safetensors
├── ComfyUI/models/vae
|        ├─Qwen-Image.safetensors  
├── ComfyUI/models/clip
|        ├──qwen_2.5_vl_7b_fp8_scaled.safetensors
├── ComfyUI/models/loras 
|        ├──Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```

# Example
![](https://github.com/smthemex/ComfyUI_GRAG_Image_Editing/blob/main/example_workflows/example.png)

### GRAGPatch parameters

- `cond_delta` — the GRAG strength. **< 1 frees the output from the reference (stronger edit);
  > 1 locks onto the reference image.** Upstream recommended sweep: `0.8 ~ 1.7`, step `0.01`,
  default `1.10`. Verified on 2511 (cfg 4.0, 24 steps): `1.70 → corr 0.995 with the input`,
  `1.10 → 0.997`, `0.90 → 0.928`, `0.80 → 0.711`.
- `cond_b` — bias (lambda) scale; upstream keeps it at `1.0` everywhere. Leave it alone.
- `img_len` — how many **trailing** image tokens form the reference group. comfy's
  `TextEncodeQwenImageEdit` resizes the reference to 1024×1024 → 4096 tokens, which is the
  default and matches upstream's hardcoded value. Use `img_len = (ref_w/16) * (ref_h/16)`
  summed over references when you deviate from that (e.g. `TextEncodeQwenImageEditPlus` with
  multiple images, or a custom non-resizing encoder). `0` = the whole image block.
- `txt_delta` / `txt_b` — same reweighting applied to the text group; upstream no-ops (1.0).
- `apply_from` / `apply_to` — sigma window (denoising progress 0→1). Defaults cover all steps,
  matching upstream.
- `verbose` — print the effective config when the graph runs.

# Citation
```
@misc{zhang2025grouprelativeattentionguidance,
      title={Group Relative Attention Guidance for Image Editing}, 
      author={Xuanpu Zhang and Xuesong Niu and Ruidong Chen and Dan Song and Jianhao Zeng and Penghui Du and Haoxiang Cao and Kai Wu and An-an Liu},
      year={2025},
      eprint={2510.24657},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.24657}, 
}
```
