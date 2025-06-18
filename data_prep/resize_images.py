import os
from PIL import Image
from tqdm import tqdm

input_dir = "raw_images"
output_dir = "resized_images"
target_size = (1024, 1024)

os.makedirs(output_dir, exist_ok=True)

for img_file in tqdm(os.listdir(input_dir)):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, img_file)
        with Image.open(img_path) as img:
            # 保持宽高比调整大小
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # 创建空白画布
            canvas = Image.new("RGB", target_size, (255, 255, 255))
            
            # 将图片粘贴到中心
            x = (target_size[0] - img.width) // 2
            y = (target_size[1] - img.height) // 2
            canvas.paste(img, (x, y))
            
            # 保存
            canvas.save(os.path.join(output_dir, img_file))
            print(f"Processed: {img_file}")