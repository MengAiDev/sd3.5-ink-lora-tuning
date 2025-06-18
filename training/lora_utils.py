import torch
import os
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor

def inject_lora_into_unet(unet, rank=8):
    """将LoRA注入UNet的注意力层"""
    # 获取所有注意力处理器
    attn_processors = unet.attn_processors
    
    # 准备LoRA处理器
    lora_attn_procs = {}
    for name in attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        
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
        if hasattr(module, "to_q_lora") and module.to_q_lora is not None:
            lora_layers[f"{name}.to_q_lora"] = module.to_q_lora.state_dict()
        if hasattr(module, "to_k_lora") and module.to_k_lora is not None:
            lora_layers[f"{name}.to_k_lora"] = module.to_k_lora.state_dict()
        if hasattr(module, "to_v_lora") and module.to_v_lora is not None:
            lora_layers[f"{name}.to_v_lora"] = module.to_v_lora.state_dict()
        if hasattr(module, "to_out_lora") and module.to_out_lora is not None:
            lora_layers[f"{name}.to_out_lora"] = module.to_out_lora.state_dict()
    
    # 保存为safetensors格式
    from safetensors.torch import save_file
    save_file(lora_layers, os.path.join(output_dir, "lora_weights.safetensors"))
    print(f"Saved LoRA weights to {output_dir}")