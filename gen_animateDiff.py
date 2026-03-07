# Two steps process. generate constrained frames, then make transition smoother between them
# Add all control maps as guide, tweak gen parameters, optionally add API to increase frames capacity
import torch
import os
from PIL import Image
import sys
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
input_depth = "depth"
input_edges = "edges"
input_flow = "flow"
input_seg = "segmentation"
output_path = "scriptOutput.gif"
resolution = (512, 384)
frames_num = 4


# load conditioning frames
depth_frames = [f for f in sorted(os.listdir(input_depth)) if f.lower().endswith((".png",".jpg",".jpeg"))][:frames_num]
edges_frames = [f for f in sorted(os.listdir(input_edges)) if f.lower().endswith((".png",".jpg",".jpeg"))][:frames_num]
flow_frames = [f for f in sorted(os.listdir(input_flow)) if f.lower().endswith((".png",".jpg",".jpeg"))][:frames_num]
seg_frames = [f for f in sorted(os.listdir(input_seg)) if f.lower().endswith((".png",".jpg",".jpeg"))][:frames_num]

depth_parsed = [
    Image.open(os.path.join(input_depth, f)).convert("L").resize(resolution).convert("RGB")
    for f in depth_frames
]
edges_parsed = [
    Image.open(os.path.join(input_edges, f)).convert("L").resize(resolution).convert("RGB")
    for f in edges_frames
]
flow_parsed = [
    Image.open(os.path.join(input_flow, f)).convert("L").resize(resolution).convert("RGB")
    for f in flow_frames
]
seg_parsed = [
    Image.open(os.path.join(input_seg, f)).convert("RGB").resize(resolution, Image.NEAREST)
    for f in seg_frames
]

control_maps = [depth_parsed, edges_parsed, seg_parsed]

# models
base_model = "Lykon/dreamshaper-8"

depth_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float32,
)

edges_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float32,
)

seg_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_seg",
    torch_dtype=torch.float32,
)

controlnet = [depth_controlnet, edges_controlnet, seg_controlnet]

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=torch.float32,
)

# prompts
# prompt = sys.argv[1]
prompt = "a blue soccer football sliding down a green inclined plane, minimal scene, fixed camera, sharp details"
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
cond_scale_depth = 1.1
cond_scale_edge = 1.1
cond_scale_seg = 1.1
num_inference_steps_1 = 24
prev_blend = 0.15

repainted = []
prev = None

with torch.inference_mode():
    for i in range(frames_num):
        depth_f = depth_parsed[i]
        edge_f = edges_parsed[i]
        seg_f = seg_parsed[i]
        init = prev if prev is not None else depth_f

        if prev is not None:
            prev = prev.convert("RGB").resize(resolution)
            depth_f = depth_f.convert("RGB").resize(prev.size)
            init = Image.blend(prev, depth_f, alpha=prev_blend)

        img = repaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init,
            control_image=[depth_f, edge_f, seg_f],
            strength=strength,
            num_inference_steps=num_inference_steps_1,
            guidance_scale=guidance_scale_1,
            controlnet_conditioning_scale=[cond_scale_depth, cond_scale_edge, cond_scale_seg],
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

inject_flow = [
    Image.blend(repainted[i].convert("RGB"), flow_parsed[i].resize(repainted[i].size).convert("RGB"), alpha=0.15)
    if i < len(flow_parsed) else repainted[i]
    for i in range(len(repainted))
]

# generation function
with torch.inference_mode():
    result = coherence_pipeline(
        video=inject_flow,
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
