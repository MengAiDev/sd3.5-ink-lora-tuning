import os
import torch
from diffusers import UNet2DConditionModel # type: ignore
from diffusers.models.attention_processor import LoRAAttnProcessor, AttnProcessor
from safetensors.torch import save_file

def inject_lora_into_unet(unet, rank=8):
    """将LoRA注入UNet的注意力层"""
    # 保存原始处理器
    unet.original_attn_processors = unet.attn_processors
    
    # 准备LoRA处理器
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            continue
        
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )
    
    # 注入LoRA处理器
    unet.set_attn_processor(lora_attn_procs)
    return unet

def save_lora_weights(unet, output_dir):
    """保存LoRA权重"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有LoRA层
    lora_layers = {}
    for name, module in unet.named_modules():
        if isinstance(module, LoRAAttnProcessor):
            # 保存LoRA权重
            lora_layers[f"{name}.to_q_lora.down.weight"] = module.to_q_lora.down.weight
            lora_layers[f"{name}.to_q_lora.up.weight"] = module.to_q_lora.up.weight
            lora_layers[f"{name}.to_k_lora.down.weight"] = module.to_k_lora.down.weight
            lora_layers[f"{name}.to_k_lora.up.weight"] = module.to_k_lora.up.weight
            lora_layers[f"{name}.to_v_lora.down.weight"] = module.to_v_lora.down.weight
            lora_layers[f"{name}.to_v_lora.up.weight"] = module.to_v_lora.up.weight
            lora_layers[f"{name}.to_out_lora.down.weight"] = module.to_out_lora.down.weight
            lora_layers[f"{name}.to_out_lora.up.weight"] = module.to_out_lora.up.weight
    
    # 保存为safetensors格式
    save_file(lora_layers, os.path.join(output_dir, "lora_weights.safetensors"))
    print(f"✅ Saved LoRA weights to {output_dir}")