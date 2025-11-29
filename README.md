# ComfyUI_GRAG_Image_Editing
[GRAG-Image-Editing](https://github.com/little-misfit/GRAG-Image-Editing) : Group-Relative Attention Guidance for Image Editing,you can try it in comfyUI 

1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_GRAG_Image_Editing
```
2.requirements  
----
* 不装也行，没什么需求，diffuser版本高点
```
pip install -r requirements.txt
```
3.Model
----

* gguf or transformer  [smthem/Qwen-Image-GGUF](https://huggingface.co/smthem/Qwen-Image-GGUF/tree/main)  or other or comfy-org optional/随便哪个gguf或者comfyUI官方的transformer  
* comfyUI normal： qwen-image  vae and qwen_2.5_vl_7b   
* lora, lightx2v 8step lora# 千问edit加速  
```
├── ComfyUI/models/gguf # or transformer
|     ├── Qwen-Image-BF16.gguf # or Q8
├── ComfyUI/models/diffusion_models # or gguf
|     ├── Qwen-Image-BF16..safetensors # or e4m3fn
├── ComfyUI/models/vae
|        ├─Qwen-Image.safetensors  # rename it 换个名字
├── ComfyUI/models/clip
|        ├──qwen_2.5_vl_7b_fp8_scaled.safetensors
├── ComfyUI/models/loras 
|        ├──Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors

```


# Example
![](https://github.com/smthemex/ComfyUI_GRAG_Image_Editing/blob/main/example_workflows/example.png)

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
