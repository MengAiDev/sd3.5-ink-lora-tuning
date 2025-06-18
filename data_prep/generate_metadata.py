import json
import os
from pathlib import Path
import random

image_dir = "resized_images"
output_file = "metadata.jsonl"

# 水墨画风格描述模板
ink_styles = [
    "Chinese ink wash painting",
    "Traditional ink and wash style",
    "Monochrome ink painting",
    "Splashed ink technique",
    "Minimalist ink brushwork"
]

artists = [
    "Qi Baishi", "Xu Beihong", "Wu Guanzhong", 
    "Li Keran", "Song Dynasty master"
]

with open(output_file, "w") as f:
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 生成多样化描述
            style = random.choice(ink_styles)
            artist = random.choice(artists)
            subject = Path(img_file).stem.replace("_", " ")
            
            prompt = f"{style} of {subject}, {artist} style, brush strokes visible, empty space"
            
            # 写入JSONL
            metadata = {
                "file_name": img_file,
                "text": prompt
            }
            f.write(json.dumps(metadata) + "\n")

print(f"Generated metadata for {len(os.listdir(image_dir))} images")