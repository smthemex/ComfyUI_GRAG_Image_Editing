import os
import torch
import sys
from termcolor import colored
from diffusers import QwenImageEditPipeline,QuantoConfig
from .hacked_models.scheduler import FlowMatchEulerDiscreteScheduler
from .hacked_models.pipeline import QwenImageEditPipeline
from .hacked_models.models import QwenImageTransformer2DModel
from .hacked_models.pipeline_plus import QwenImageEditPlusPipeline
import sys
from .hacked_models.utils import *
from contextlib import contextmanager
from accelerate import init_empty_weights
import sys 
from .util import load_checkpoint_and_dispatch_
from accelerate.big_modeling import dispatch_model,get_balanced_memory,infer_auto_device_map
@contextmanager
def temp_patch_module_attr(module_name: str, attr_name: str, new_obj):
    mod = sys.modules.get(module_name)
    if mod is None:
        yield
        return
    had = hasattr(mod, attr_name)
    orig = getattr(mod, attr_name, None)
    setattr(mod, attr_name, new_obj)
    try:
        yield
    finally:
        if had:
            setattr(mod, attr_name, orig)
        else:
            try:
                delattr(mod, attr_name)
            except Exception:
                pass

def load_model(gguf_path,unet_path,node_path):
    plus_mode=False
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit/scheduler"),torch_dtype=torch.bfloat16,)
    if  gguf_path is not None:
        if "2509" in gguf_path.lower() or  "plus" in gguf_path.lower():
            plus_mode=True
        from diffusers import  GGUFQuantizationConfig
        with temp_patch_module_attr("diffusers", "QwenImageTransformer2DModel", QwenImageTransformer2DModel):
            transformer = QwenImageTransformer2DModel.from_single_file(
                gguf_path,
                config=os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit/transformer"),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
    else:
        if "2509" in unet_path.lower() or  "plus" in unet_path.lower():
            plus_mode=True
        
        try:
            print("loading from  single file safetensors")
            with temp_patch_module_attr("diffusers", "QwenImageTransformer2DModel", QwenImageTransformer2DModel):
                transformer = QwenImageTransformer2DModel.from_single_file(unet_path,config=os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit/transformer"),torch_dtype=torch.bfloat16,)
        except:
            print("loading from single file failed, try to load from single file from dict")
            from safetensors.torch import load_file
            sd=load_file(unet_path)
            sd=replace_key(sd)
            with init_empty_weights():
                with temp_patch_module_attr("diffusers", "QwenImageTransformer2DModel", QwenImageTransformer2DModel):
                    unet_config = QwenImageTransformer2DModel.load_config(os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit/transformer/config.json"))
                    transformer = QwenImageTransformer2DModel.from_config(unet_config).to(torch.bfloat16)
            model_state_dict = transformer.state_dict()
            expected_keys = set(model_state_dict.keys())
            del transformer
            filtered_sd = {k: v for k, v in sd.items() if k in expected_keys}
            # 检查是否有缺失的键
            missing_keys = expected_keys - set(sd.keys())
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            del sd  
            with temp_patch_module_attr("diffusers", "QwenImageTransformer2DModel", QwenImageTransformer2DModel):
                transformer = QwenImageTransformer2DModel.from_single_file(filtered_sd,config=os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit/transformer"),torch_dtype=torch.bfloat16,)
            del filtered_sd
        
    if plus_mode:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit"), scheduler = scheduler,vae=None,text_encoder=None,transformer=transformer,torch_dtype=torch.bfloat16,)
    else:
        pipeline = QwenImageEditPipeline.from_pretrained(os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit"), scheduler = scheduler,vae=None,text_encoder=None,transformer=transformer,torch_dtype=torch.bfloat16,)   
   
    return pipeline

def replace_key(t_state_dict):
    return {k.replace("model.diffusion_model.", "", 1): v for k, v in t_state_dict.items()}

def inference(pipeline,positive,negative,num_inference_steps,seed,true_cfg_scale,cond_b,cond_delta):
    
    seed_everything(seed)
    pipeline.set_progress_bar_config(disable=None)

    inputs = {
        "image":None,
        "prompt": None,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": None,
        "num_inference_steps": num_inference_steps,
        "prompt_embeds": positive[0][0], 
        "negative_prompt_embeds": negative[0][0],
        "image_latents":positive[0][1].get("reference_latents",None) , 
        "return_dict": False,
        "grag_scale":[((512,1.0,1.0),(4096,cond_b,cond_delta))]*60,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        image,x0_images,saved_outputs = output

    #image[0].save(os.path.join(out_path,f"{args.image_path.split('/')[-1]}_cond_b-{args.cond_b}_cond_delta-{args.cond_delta}.jpg"))
    return image,x0_images



