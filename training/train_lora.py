import yaml
import torch
import os
import numpy as np

from accelerate import Accelerator
from datasets import load_dataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from .lora_utils import inject_lora_into_unet, save_lora_weights
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

# 加载配置
with open("configs/train_config.yaml") as f:
    config = yaml.safe_load(f)

# 初始化加速器
accelerator = Accelerator(
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    mixed_precision=config["mixed_precision"],
    log_with="tensorboard",
    project_dir=config["output_dir"]
)

# 准备数据集
dataset = load_dataset("imagefolder", data_dir=config["dataset_path"])
train_dataset = dataset["train"] # type: ignore

# 加载UNet模型
unet = UNet2DConditionModel.from_pretrained(
    config["model"],
    subfolder="unet",
    torch_dtype=torch.float16
)

# 注入LoRA
unet = inject_lora_into_unet(unet, rank=config["lora_rank"])

# 冻结原始参数（只训练LoRA层）
for param in unet.parameters(): # type: ignore
    param.requires_grad = False
for param in unet.attn_processors.parameters(): # type: ignore
    param.requires_grad = True

# 准备文本编码器和VAE（仅用于前向传播）
text_encoder = CLIPTextModelWithProjection.from_pretrained(
    config["model"], subfolder="text_encoder", torch_dtype=torch.float16
)
tokenizer = CLIPTokenizer.from_pretrained(
    config["model"], subfolder="tokenizer"
)
vae = AutoencoderKL.from_pretrained(
    config["model"], subfolder="vae", torch_dtype=torch.float16
)

# 噪声调度器
noise_scheduler = DDPMScheduler.from_pretrained(config["model"], subfolder="scheduler")

# 数据加载器
def collate_fn(examples):
    images = [example["image"] for example in examples]
    texts = [example["text"] for example in examples]
    
    # 图像预处理
    pixel_values = torch.stack([
        torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0 
        for image in images
    ])
    
    # 文本编码
    text_inputs = tokenizer(
        texts, 
        max_length=tokenizer.model_max_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    
    return {"pixel_values": pixel_values, "input_ids": text_inputs.input_ids}

train_dataloader = DataLoader(
    train_dataset, # type: ignore
    batch_size=config["batch_size"], 
    shuffle=True,
    collate_fn=collate_fn
)

# 优化器和学习率调度器
optimizer = torch.optim.AdamW(
    unet.attn_processors.parameters(),  # 只优化LoRA层 # type: ignore
    lr=config["learning_rate"]
)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config["lr_warmup_steps"],
    num_training_steps=config["max_train_steps"]
)

# 准备组件
unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler
)

# 训练循环
global_step = 0
progress_bar = tqdm(range(config["max_train_steps"]))

for epoch in range(10):  # 固定epoch数
    for batch in train_dataloader:
        # 仅训练LoRA层
        unet.train()
        
        with accelerator.accumulate(unet):
            # 获取文本嵌入
            text_embeddings = text_encoder(batch["input_ids"])[0]
            
            # 编码图像到潜在空间
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), device=latents.device
            ).long()
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 预测噪声
            noise_pred = unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states=text_embeddings
            ).sample
            
            # 计算损失
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # 反向传播
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # 更新进度
        global_step += 1
        progress_bar.update(1)
        accelerator.log({"loss": loss.item()}, step=global_step)
        
        # 定期保存
        if global_step % 200 == 0:
            accelerator.wait_for_everyone()
            save_lora_weights(accelerator.unwrap_model(unet), 
                             os.path.join(config["output_dir"], f"checkpoint-{global_step}"))
        
        if global_step >= config["max_train_steps"]:
            break

# 最终保存
accelerator.wait_for_everyone()
save_lora_weights(accelerator.unwrap_model(unet), os.path.join(config["output_dir"], "final"))
accelerator.end_training()