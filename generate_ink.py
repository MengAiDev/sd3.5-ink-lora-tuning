import torch
import diffusers
from modelscope import StableDiffusion3Pipeline
from diffusers.utils import logging

# 启用详细日志（可选）
logging.set_verbosity_info()

# 1. 配置参数
MODEL_NAME = "AI-ModelScope/stable-diffusion-3.5-medium"  # 基础模型（需与微调时一致）
LORA_PATH = "./sd3_model"                      # LoRA权重所在目录
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
PROMPT = "Chinese ink painting, {subject: mountains|bamboo|birds|scholars},  brushstroke texture, ink wash gradients, empty space,  subtle mist effect, aged paper texture, {style: Song Dynasty|Ukiyo-e}, seal stamp, calligraphic inscription  "  # 替换为你的触发词+提示
NEGATIVE_PROMPT = "photorealistic, 3D render, oil painting, bright colors,  hyperdetailed, Western art, anime, digital art  "
OUTPUT_FILE = "lora_output.png"

# 2. 加载基础模型 + LoRA
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # 半精度节省显存
    safety_checker=None,        # 可选：禁用安全检查（加快推理）
).to("cuda")

# 注入LoRA权重（注意：sd3_model是目录，pytorch_lora_weights.safetensors是文件名）
pipe.load_lora_weights(LORA_PATH, weight_name=LORA_WEIGHT_NAME)

# 可选：调整LoRA强度（默认1.0，范围0~1.5）
lora_scale = 0.8
cross_attention_kwargs = {"scale": lora_scale}

# 3. 生成图像
generator = torch.Generator("cuda").manual_seed(42)  # 固定随机种子（可选）

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=30,
    guidance_scale=7.5,
    width=512,
    height=512,
    generator=generator,
    cross_attention_kwargs=cross_attention_kwargs  # 控制LoRA强度
).images[0]

# 4. 保存结果
image.save(OUTPUT_FILE)
print(f"Generated image saved to {OUTPUT_FILE}")
