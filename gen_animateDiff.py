# Going for two steps process. generate constrained frames, thenmake transition smoother between them
import torch
import os
from PIL import Image
from diffusers import (
    ControlNetModel,
    MotionAdapter,
    StableDiffusionControlNetImg2ImgPipeline,
    AnimateDiffVideoToVideoPipeline,
    DDIMScheduler,
)
from diffusers.utils import export_to_gif


# setting backend
device = "mps" if torch.backends.mps.is_available() else "cpu"

# fixed seed for deterministic outputs
generator = torch.Generator(device="cpu").manual_seed(42)

# folders
input_maps = "depth"
output_path = "scriptOutput.gif"
resolution = (540, 540)
frames_num = 4


# load conditioning frames
input_frames = [f for f in sorted(os.listdir(input_maps)) if f.lower().endswith((".png",".jpg",".jpeg"))][:frames_num]
parsed_frames = [
    Image.open(os.path.join(input_maps, f)).convert("L").resize(resolution).convert("RGB")
    for f in input_frames
]

# models
base_model = "Lykon/dreamshaper-8"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float32,
)

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float32,
)

# prompts
prompt = "a single red cube at the top of the screen, falling down above a blue inclined plane, minimal scene, fixed camera, sharp details"
negative_prompt = "duplicate, extra objects, blur, motion blur, flicker, noise, artifacts, text"


# Step 1: Repaint
repaint_pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None,
    feature_extractor=None,
).to(device)

repaint_pipeline.enable_attention_slicing()
repaint_pipeline.enable_vae_slicing()
repaint_pipeline.scheduler = DDIMScheduler.from_config(
    repaint_pipeline.scheduler.config
)

# diffusion knobs pipeline 1
strength = 0.45
guidance_scale_1 = 6.5
controlnet_conditioning_scale_1 = 1.1
num_inference_steps_1 = 24
prev_blend = 0.15

repainted = []
prev = None

with torch.inference_mode():
    for c in parsed_frames:
        init = prev if prev is not None else c
        if prev is not None:
            c = c.resize(prev.size).convert("RGB")
            prev = prev.convert("RGB")
            init = Image.blend(prev, c, alpha=prev_blend)

        img = repaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init,
            control_image=c,
            strength=strength,
            num_inference_steps=num_inference_steps_1,
            guidance_scale=guidance_scale_1,
            controlnet_conditioning_scale=controlnet_conditioning_scale_1,
            generator=generator,
        ).images[0]

        repainted.append(img)
        prev = img



# Step 2: temporal coherence between frames
coherence_pipeline = AnimateDiffVideoToVideoPipeline.from_pretrained(
    base_model,
    motion_adapter=adapter,
    torch_dtype=torch.float32,
).to(device)

coherence_pipeline.enable_attention_slicing()
coherence_pipeline.enable_vae_slicing()
coherence_pipeline.scheduler = DDIMScheduler.from_config(
    coherence_pipeline.scheduler.config
)

# diffusion knobs pipeline 2
guidance_scale_2 = 5.5
strength_2 = 0.35
num_inference_steps_2 = 24

# generation function
with torch.inference_mode():
    result = coherence_pipeline(
        video=repainted,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength_2,
        num_inference_steps=num_inference_steps_2,
        guidance_scale=guidance_scale_2,
        generator=generator,
    )
output_frames = result.frames[0]

# save to file
export_to_gif(output_frames, output_path, fps=8)
print("Output:", output_path)