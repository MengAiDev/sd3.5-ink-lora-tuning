
# **🎨 SD3 Chinese Ink Painting Model – Fine-Tuned for Traditional Aesthetics**  
*"Where AI Meets Classical Oriental Art"*  

This model is a **specialized fine-tuned version of Stable Diffusion 3**, meticulously trained to emulate **authentic Chinese ink wash painting (水墨画, *shuǐmò huà*)**. It captures the essence of **monochrome brushwork, poetic emptiness, and dynamic ink diffusion**, making it ideal for generating **landscapes, calligraphy, flora/fauna, and figurative art** in the style of **Song Dynasty masters, Bada Shanren, or Qi Baishi**.  Open source at: https://modelscope.cn/models/MengAiDev/SD3FinetuneChineseInk/

---  

## **⚙️ Recommended Inference Settings (StableSwarmUI)**  

For optimal results, use the following parameters:  

### **📌 Core Parameters**  
| Setting | Recommended Value | Notes |  
|---------|------------------|-------|  
| **Base Model** | `sd3_ink_wash.safetensors` | Primary checkpoint |  
| **Resolution** | `1024x1024` (standard) <br> `768x1536` (scroll-style) | Higher res improves brush detail |  
| **Sampler** | `DPM++ 3M SDE` (Karras) | Best for smooth gradients |  
| **Steps** | `28-35` | Balances detail and speed |  
| **CFG Scale** | `5.5-7.0` | Avoid over-saturation |  
| **Clip Skip** | `2` | Enhances stylistic adherence |  

### **🎨 Key Prompt Engineering**  
#### **Positive Prompt (Mandatory Elements)**  
```  
Chinese ink painting, {subject: mountains|bamboo|birds|scholars},  
brushstroke texture, ink wash gradients, empty space (留白),  
subtle mist effect, aged paper texture, {style: Song Dynasty|Ukiyo-e},  
seal stamp (印章), calligraphic inscription  
```  
#### **Negative Prompt (Avoid Western Styles)**  
```  
photorealistic, 3D render, oil painting, bright colors,  
hyperdetailed, Western art, anime, digital art  
```  

---  

## **🖌️ Advanced Techniques**  
### **1. Dynamic Ink Control**  
- Use **`ink density:0.2~1.0`** in X/Y plots to compare light (*feibai*) vs. heavy (*nongmo*) ink.  
- Add **`<lora:InkFlow_v2:0.6>`** for enhanced brushstroke realism.  

### **2. Scroll & Panorama Generation**  
- Enable **Tiled Diffusion** (in SwarmUI’s Extras) for **long scroll formats** (e.g., 512x2048).  
- Trigger **"Along the River During Qingming Festival"** style with:  
  ```  
  ancient Chinese cityscape, bustling riverbank, ink scroll panorama,  
  detailed figures, seasonal atmosphere  
  ```  

### **3. Style Presets (SwarmUI Integration)**  
- Load **`ChineseInk.json`** from the Style Gallery for:  
  - **Gongbi (工笔)** (detailed realism)  
  - **Xieyi (写意)** (freehand abstraction)  
  - **Splashed Ink (泼墨)** (expressive blots)  

---  

## **⚠️ Known Limitations**  
- Struggles with **complex modern scenes** (stick to classical themes).  
- **Human faces** may appear overly stylized (use `portrait` with caution).  
- For **color ink (淡彩)**, add `slight watercolor tones` but avoid vibrant hues.  
