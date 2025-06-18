import torch
import os

from modelscope import StableDiffusion3Pipeline  # type: ignore
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from training.lora_utils import inject_lora_into_unet

def load_lora_weights(unet, lora_path):
    """加载LoRA权重"""
    from safetensors.torch import load_file
    lora_state_dict = load_file(os.path.join(lora_path, "lora_weights.safetensors"))
    
    # 应用LoRA权重
    for name, param in unet.named_parameters():
        for lora_name in lora_state_dict.keys():
            if lora_name in name:
                # 创建LoRA层（如果不存在）
                if not hasattr(param, "lora_layer"):
                    # 这里需要根据层类型创建对应的LoRA层
                    # 简化实现，实际需要更复杂的映射
                    setattr(param, "lora_layer", torch.nn.Linear(
                        param.shape[1], param.shape[0], bias=False
                    ).to(param.device))
                
                # 加载权重
                getattr(param, "lora_layer").weight.data = lora_state_dict[lora_name]
                break
    
    return unet

def generate_ink_painting(
    prompt, 
    lora_path, 
    model_name="stabilityai/stable-diffusion-3-medium-diffusers",
    negative_prompt=None,
    height=1024,
    width=1024,
    num_steps=30,
    guidance_scale=7.0,
    lora_scale=0.8,
    seed=42
):
    # 加载基础模型
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    # 注入LoRA
    pipe.unet = inject_lora_into_unet(pipe.unet)
    
    # 加载LoRA权重
    pipe.unet = load_lora_weights(pipe.unet, lora_path)
    
    # 设置采样器
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # 生成图像
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "photorealistic, oil painting, bright colors, signature, text",
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        cross_attention_kwargs={"scale": lora_scale}  # LoRA权重强度
    ).images[0]
    
    return image

# 使用方式保持不变