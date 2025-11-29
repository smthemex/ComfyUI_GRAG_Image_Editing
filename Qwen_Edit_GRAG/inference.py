import os
import torch
import sys
from termcolor import colored
from diffusers import QwenImageEditPipeline
from .hacked_models.scheduler import FlowMatchEulerDiscreteScheduler
from .hacked_models.pipeline import QwenImageEditPipeline
from .hacked_models.models import QwenImageTransformer2DModel
import sys
from .hacked_models.utils import *
from contextlib import contextmanager
import sys 
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

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit/scheduler"),torch_dtype=torch.bfloat16,)
    if  gguf_path is not None:
        from diffusers import  GGUFQuantizationConfig
        with temp_patch_module_attr("diffusers", "QwenImageTransformer2DModel", QwenImageTransformer2DModel):
            transformer = QwenImageTransformer2DModel.from_single_file(
                gguf_path,
                config=os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit/transformer"),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
    else:
        try:
            transformer = QwenImageTransformer2DModel.from_single_file(gguf_path,config=os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit/transformer"),torch_dtype=torch.bfloat16,)
        except:
            print("loading from safetensors")
        from safetensors.torch import load_file
        t_state_dict=load_file(unet_path)
        new_dict=replace_key(t_state_dict)
        with temp_patch_module_attr("diffusers", "QwenImageTransformer2DModel", QwenImageTransformer2DModel):
            unet_config = QwenImageTransformer2DModel.load_config(os.path.join(node_path, "Qwen_Edit_GRAG/Qwen-Image-Edit/transformer/config.json"))
            transformer = QwenImageTransformer2DModel.from_config(unet_config).to(torch.bfloat16)
        transformer.load_state_dict(new_dict, strict=False)
        del t_state_dict,new_dict
   
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
        "prompt_embeds": positive[0][0], #pooled_prompt_embeds=positive[0][1].get("pooled_output")
        "negative_prompt_embeds": negative[0][0],
        "image_latents":positive[0][1].get("ref_latents",None) , 
        "return_dict": False,
        "grag_scale":[((512,1.0,1.0),(4096,cond_b,cond_delta))]*60,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        image,x0_images,saved_outputs = output

    #image[0].save(os.path.join(out_path,f"{args.image_path.split('/')[-1]}_cond_b-{args.cond_b}_cond_delta-{args.cond_delta}.jpg"))
    return image,x0_images





#parser = argparse.ArgumentParser() 

# parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit")
# parser.add_argument("--image_path", type=str, required=True)
# parser.add_argument("--edit_prompt", type=str, required=True)
# parser.add_argument("--out_path", type=str, default='./results')
# parser.add_argument("--cond_b", type=float, required=True)
# parser.add_argument("--cond_delta", type=float, required=True)



# args = parser.parse_args()

# scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
#     os.path.join(args.model_path, "scheduler"),
#     torch_dtype=torch.bfloat16,
# )


# transformer = QwenImageTransformer2DModel.from_pretrained(
#     os.path.join(args.model_path, "transformer"),
#     torch_dtype=torch.bfloat16,
# )

# pipeline = QwenImageEditPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
#                                             scheduler = scheduler,
#                                             transformer=transformer,
#                                             )
# print("pipeline loaded")
# pipeline.to(torch.bfloat16)
# pipeline.to("cuda")
# pipeline.set_progress_bar_config(disable=None)


# out_path = args.out_path
# os.makedirs(out_path,exist_ok=True)

# print(colored(out_path,color = "green"))


# editing_instruction = args.edit_prompt
# input_image = Image.open(args.image_path).convert('RGB').resize((1024,1024))

# os.makedirs(os.path.join(out_path),exist_ok=True)

# prompt = editing_instruction
# inputs = {
#     "image": input_image,
#     "prompt": prompt,
#     "generator": torch.manual_seed(42),
#     "true_cfg_scale": 4.0,
#     "negative_prompt": " ",
#     "num_inference_steps": 24,
#     "return_dict": False,
#     "grag_scale":[((512,1.0,1.0),(4096,args.cond_b,args.cond_delta))]*60,
# }

# with torch.inference_mode():
#     output = pipeline(**inputs)
#     image,x0_images,saved_outputs = output

# image[0].save(os.path.join(out_path,f"{args.image_path.split('/')[-1]}_cond_b-{args.cond_b}_cond_delta-{args.cond_delta}.jpg"))
    