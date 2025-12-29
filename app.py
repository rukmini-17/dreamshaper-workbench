import gradio as gr
import torch
import time
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler

# --- SETUP: Model Loading (CPU Optimized) ---
model_id = "Lykon/dreamshaper-8"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

print("⏳ Loading Pipeline...")

# 1. Check if we are on a Mac (MPS) or standard CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # CPU needs float32

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_scribble", 
    torch_dtype=dtype
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, 
    controlnet=controlnet, 
    torch_dtype=dtype, 
    safety_checker=None
).to(device)

pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.fuse_lora()
print(f"✅ Model Loaded on {device.upper()}.")

# --- STYLE TEMPLATES ---
style_templates = {
    "Cinematic": "cinematic lighting, shallow depth of field, 8k, realistic texture, highly detailed, {prompt}",
    "3D Render": "3d render, unreal engine 5, octane render, bright colors, smooth, {prompt}",
    "Anime": "anime style, studio ghibli, vibrant colors, cell shaded, highly detailed, {prompt}",
    "Sketch": "charcoal sketch, black and white, rough lines, artistic, {prompt}",
    "Neon/Cyberpunk": "cyberpunk, neon lights, dark background, synthwave, futuristic, {prompt}"
}

# --- INFERENCE FUNCTION (No @spaces decorator needed) ---
def process_sketch(user_sketch, user_prompt, style_choice, seed, guidance, compare_mode):
    if user_sketch is None:
        return None, ""
    
    start_time = time.time()
    
    if isinstance(user_sketch, dict):
        image_input = user_sketch.get("composite") or user_sketch.get("image")
    else:
        image_input = user_sketch
        
    image_input = image_input.convert("RGB").resize((512, 512))
    
    if compare_mode:
        target_styles = list(style_templates.keys())
    else:
        target_styles = [style_choice]

    gallery_results = []
    
    if seed == -1:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    
    for style in target_styles:
        final_prompt = style_templates[style].format(prompt=user_prompt)
        generator = torch.manual_seed(int(seed))

        output = pipe(
            prompt=final_prompt,
            image=image_input,
            num_inference_steps=6,
            guidance_scale=guidance,
            controlnet_conditioning_scale=0.8, 
            cross_attention_kwargs={"scale": 1.0},
            generator=generator
        ).images[0]
        
        gallery_results.append((output, style))
    
    total_time = time.time() - start_time
    
    metrics_html = f"""
    <div style='font-size: 12px; color: #555; font-family: monospace; border-top: 1px solid #eee; padding-top: 8px;'>
    ⏱️ Total Time: <b>{total_time:.3f}s</b> (Running on Free CPU Tier) &nbsp;|&nbsp; 
    🌱 Seed Used: <b>{seed}</b> &nbsp;|&nbsp; 
    🖼️ Variations: <b>{len(gallery_results)}</b>
    </div>
    """
    
    return gallery_results, metrics_html

# --- UI LAYOUT ---
css = """
#col-container {max_width: 1000px; margin-left: auto; margin-right: auto;}
#gallery { height: auto !important; overflow: visible !important; }
#gallery img { 
    max-height: 400px !important; 
    width: auto !important; 
    object-fit: contain; 
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ DreamShaper Interactive Workbench")
    gr.Markdown("Real-time generative design tool. Draw once, explore multiple styles instantly.")
    gr.Markdown("*Note: This demo is running on a free CPU tier. Generation may take ~15-30 seconds. See GitHub for GPU performance.*")
    
    with gr.Row(elem_id="col-container"):
        # Left Column (Input)
        with gr.Column(scale=4, min_width=300):
            sketch_pad = gr.Sketchpad(
                label="Input Structure", 
                type="pil", 
                height=400,
                width=400,
                layers=False
            )
            
            with gr.Group():
                user_prompt = gr.Textbox(label="Prompt", placeholder="Describe your scene...")
                
                with gr.Row():
                    style = gr.Dropdown(choices=list(style_templates.keys()), value="Cinematic", label="Primary Style")
                    compare_chk = gr.Checkbox(label="Compare All Styles", value=False, info="Generates 5 variations grid")
            
            with gr.Accordion("Advanced Settings", open=False):
                slider_guidance = gr.Slider(
                    0.5, 3.0, value=1.5, step=0.1, 
                    label="Guidance Intensity",
                    info="Lower = More Creative/Abstract | Higher = Stricter adherence to prompt"
                )
                slider_seed = gr.Number(value=-1, label="Seed (-1 for Random)", precision=0)

            btn = gr.Button("✨ Render", variant="primary")
        
        # Right Column (Output)
        with gr.Column(scale=5):
            result_gallery = gr.Gallery(
                label="Rendered Variations", 
                show_label=True, 
                elem_id="gallery", 
                columns=3,     
                rows=None,
                object_fit="contain",
                height="auto",
                preview=True
            )
            metrics_display = gr.Markdown()

    btn.click(
        fn=process_sketch, 
        inputs=[sketch_pad, user_prompt, style, slider_seed, slider_guidance, compare_chk], 
        outputs=[result_gallery, metrics_display]
    )

if __name__ == "__main__":
    demo.launch()