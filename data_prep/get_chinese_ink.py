import argparse
import os
import requests
import time
import random
from bs4 import BeautifulSoup
from pathlib import Path

def download_ink_wash_paintings(output_dir, num_images=600):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置Bing搜索URL和请求头
    base_url = "https://www.bing.com/images/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    downloaded = 0
    page = 0
    
    print(f"Starting download of {num_images} ink wash paintings...")
    
    while downloaded < num_images:
        # 构建搜索URL (使用不同的水墨画相关关键词)
        keywords = ["水墨画", "ink wash painting", "Chinese ink painting", "山水画"]
        query = random.choice(keywords)
        params = {
            "q": query,
            "qft": "+filterui:photo-photo",
            "form": "IRFLTR",
            "first": page * 150 + 1
        }
        
        try:
            # 获取搜索结果页面
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            
            # 解析HTML获取图片URL
            soup = BeautifulSoup(response.text, 'html.parser')
            image_elements = soup.select('.mimg')
            
            if not image_elements:
                print("No more images found. Exiting.")
                break
            
            for img in image_elements:
                if downloaded >= num_images:
                    break
                    
                try:
                    # 获取实际图片URL
                    img_url = img.get('src')
                    if not img_url or not img_url.startswith('http'): # type: ignore
                        img_url = img.get('data-src', '')
                    
                    if img_url:
                        # 下载图片
                        img_data = requests.get(img_url, headers=headers, timeout=10) # type: ignore
                        img_data.raise_for_status()
                        
                        # 保存图片
                        ext = img_url.split('.')[-1].split('?')[0].lower() # type: ignore
                        if ext not in ['jpg', 'jpeg', 'png', 'gif']:
                            ext = 'jpg'
                            
                        filename = f"ink_wash_{downloaded+1:04d}.{ext}"
                        save_path = os.path.join(output_dir, filename)
                        
                        with open(save_path, 'wb') as f:
                            f.write(img_data.content)
                            
                        downloaded += 1
                        print(f"Downloaded: {downloaded}/{num_images} - {img_url[:60]}...")
                        
                        # 随机延时避免被屏蔽
                        time.sleep(random.uniform(0.2, 0.8))
                
                except Exception as e:
                    print(f"Error downloading image: {str(e)}")
                    continue
            
            page += 1
            print(f"Moving to next page: {page}")
            
        except Exception as e:
            print(f"Error fetching search page: {str(e)}")
            time.sleep(5)
    
    print(f"Download completed. Total images: {downloaded}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download ink wash paintings from Bing')
    parser.add_argument('-o', '--output', type=str, default='ink_paintings',
                        help='Output directory for images')
    parser.add_argument('-n', '--num', type=int, default=600,
                        help='Number of images to download')
    
    args = parser.parse_args()
    
    download_ink_wash_paintings(args.output, args.num)