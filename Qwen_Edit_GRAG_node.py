 # !/usr/bin/env python
# -*- coding: UTF-8 -*-
import math
import numpy as np
import torch
import os
from diffusers.hooks import apply_group_offloading
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
import comfy.model_management as mm
from .model_loader_utils import tensor2list,nomarl_upscale,get_emb_data
from .Qwen_Edit_GRAG.inference import load_model,inference
from .Qwen_Edit_GRAG.hacked_models.scheduler import FlowMatchEulerDiscreteScheduler
import node_helpers
MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")


weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)

folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class Qwen_Edit_GRAG_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="Qwen_Edit_GRAG_SM_Model",
            display_name="Qwen_Edit_GRAG_SM_Model",
            category="Qwen_Edit_GRAG",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf") ),
            ],
            outputs=[
                io.Custom("Qwen_Edit_GRAG_SM_Model").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf) -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
        gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
        pipeline = load_model(gguf_path,dit_path,node_cr_path)
        return io.NodeOutput(pipeline)

class Qwen_Edit_GRAG_SM_Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Qwen_Edit_GRAG_SM_Encode",
            display_name="Qwen_Edit_GRAG_SM_Encode",
            category="Qwen_Edit_GRAG",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Int.Input("width", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=1024, min=256, max=nodes.MAX_RESOLUTION,step=16,display_mode=io.NumberDisplay.number),
                io.String.Input("pos_text", multiline=True,default=",best"),
                io.String.Input("neg_text", multiline=True,default="bad anatomy, bad hands, missing fingers, extra fingers,three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn,amputation, disconnected limbs"),
                
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                ],
        )
    @classmethod
    def execute(cls, clip, vae,image,width,height,pos_text,neg_text,) -> io.NodeOutput:

        tensor_list=tensor2list(image,width,height)
        pli_image=nomarl_upscale(image,width,height) if isinstance(image,torch.Tensor) else None

        postive,ref_latents=get_emb_data(clip,vae,pos_text,tensor_list,)
        negative,_=get_emb_data(clip,vae,neg_text,tensor_list,ng=True,img=tensor_list[0] if tensor_list is not None else None )
        postive=node_helpers.conditioning_set_values(postive, {"ref_latents": ref_latents}) 

        # gc cf model
        cf_models=mm.loaded_models()
        try:
            for pipe in cf_models:   
                pipe.unpatch_model(device_to=torch.device("cpu"))
                print(f"Unpatching models.{pipe}")
        except: pass
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        max_gpu_memory = torch.cuda.max_memory_allocated()
        print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

        return io.NodeOutput(postive,negative)   

class Qwen_Edit_GRAG_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Qwen_Edit_GRAG_SM_KSampler",
            display_name="Qwen_Edit_GRAG_SM_KSampler",
            category="Qwen_Edit_GRAG",
            inputs=[
                io.Custom("Qwen_Edit_GRAG_SM_Model").Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras")),
                io.Int.Input("steps", default=24, min=1, max=1024,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("guidance_scale", default=4.0, min=0, max=20,step=0.01,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Float.Input("cond_b", default=1.0, min=0.01, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("cond_delta", default=1.10, min=0.01, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Int.Input("block_num", default=10, min=1, max=MAX_SEED,display_mode=io.NumberDisplay.number),
            ], 
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="latents"),
            ],
        )
    @classmethod
    def execute(cls, model,positive,negative,lora,steps,guidance_scale,seed,cond_b,cond_delta,block_num,) -> io.NodeOutput:
       
        adapter_path=folder_paths.get_full_path("loras", lora) if lora != "none" else None
        if adapter_path is not None:  
            model.load_lora_weights(adapter_path,weight_name= os.path.basename(adapter_path))
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
            model.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

         
        # apply offloading
        apply_group_offloading(model.transformer, onload_device=torch.device("cuda"), offload_type="block_level", num_blocks_per_group=block_num)
        # infer
        lat,lats=inference(model,positive,negative,steps,seed,guidance_scale,cond_b,cond_delta)
        lats=torch.cat(lats,dim=0)
        out_put={"samples":lat}
        out_puts={"samples":lats}
        return io.NodeOutput(out_put,out_puts)



from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/Qwen_Edit_GRAG_SM_Extension")
async def get_hello(request):
    return web.json_response("Qwen_Edit_GRAG_SM_Extension")

class Qwen_Edit_GRAG_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Qwen_Edit_GRAG_SM_Model,
            Qwen_Edit_GRAG_SM_Encode,
            Qwen_Edit_GRAG_SM_KSampler,
        ]
async def comfy_entrypoint() -> Qwen_Edit_GRAG_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return Qwen_Edit_GRAG_SM_Extension()



