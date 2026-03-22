# AnimateDiff pipeline, added first-frame stabilization and chunking for generating more frames.
import os
import sys
import numpy as np
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    MotionAdapter,
    StableDiffusionControlNetImg2ImgPipeline,
    AnimateDiffVideoToVideoPipeline,
    DDIMScheduler,
)
from diffusers.utils import export_to_video


# setting backend
device = "mps" if torch.backends.mps.is_available() else "cpu"
generator = torch.Generator(device="cpu").manual_seed(42)

# folders
input_depth = "depth"
input_edges = "edges"
input_flow = "flow"
input_seg = "segmentation"
output_path = "output_long_test6.mp4"
style_anchor_path = os.path.join("style_anchor", "anchor_image.png")
resolution = (512, 384)

# chunking 
chunk_length = 4
chunk_intersection = 3

# load conditioning frames
depth_files = [f for f in sorted(os.listdir(input_depth)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
edges_files = [f for f in sorted(os.listdir(input_edges)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
flow_files = [f for f in sorted(os.listdir(input_flow)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
seg_files = [f for f in sorted(os.listdir(input_seg)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

frames_num = min(len(depth_files), len(edges_files), len(seg_files))
if frames_num == 0:
    raise ValueError("No control maps frames found.")

depth_frames = depth_files[:frames_num]
edges_frames = edges_files[:frames_num]
flow_frames = flow_files[:frames_num]
seg_frames = seg_files[:frames_num]

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
obj_ids = [
    np.load(os.path.join(input_seg, os.path.splitext(f)[0] + ".npy"))
    for f in seg_frames
]
#block below selects ids of objects that are not background or plane to leave them unblended from the style-anchor frame
non_background_objects = set()

def seg_resize(seg):
    if seg.shape[:2] != (resolution[1], resolution[0]):
        seg = np.array(Image.fromarray(seg).resize(resolution, Image.NEAREST))
    return seg

obj_ids = [seg_resize(obj) for obj in obj_ids]
for obj in obj_ids:
    non_background_objects.update(int(v) for v in np.unique(obj) if v >= 1 and v != 16777215)
non_background_obj_array = np.array(sorted(non_background_objects), dtype=np.int64) if non_background_objects else None

style_anchor = Image.open(style_anchor_path).convert("RGB").resize(resolution) if os.path.exists(style_anchor_path) else None

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
# default_prompt = "a basketball ball orange colored, NBA sport style, falling down 1 meter above a 30° degrees inclined surface, minimal scene, fixed camera, empty background."
default_prompt = "a soccer football, falling down 1 meter above an inclined surface, minimal scene, fixed camera, empty background."

prompt = sys.argv[1] if len(sys.argv) > 1 else default_prompt
negative_prompt = "duplicate, extra objects, blur, inconsistent, motion blur, flicker, noise, artifacts, text, unstable texture"


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

strength = 0.33
guidance_scale_1 = 6.0
cond_scale_depth = 1.3
cond_scale_edge = 1.2
cond_scale_seg = 1.2
num_inference_steps_1 = 20
prev_blend = 0.11

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
            depth_f = depth_f.convert("RGB").resize(resolution)
            init = Image.blend(prev, depth_f, alpha=prev_blend)

            if style_anchor is not None and non_background_obj_array is not None:
                obj_id = obj_ids[i]
                anchor_area = ~np.isin(obj_id, non_background_obj_array)
                anchor_area = Image.fromarray((anchor_area * 255).astype("uint8")).convert("L").resize(resolution, Image.NEAREST)
                init = Image.composite(style_anchor, init, anchor_area)

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
        if style_anchor is None:
            style_anchor = img.convert("RGB").resize(resolution)


# free stage 1 memory before stage 2
del repaint_pipeline
del controlnet
del depth_controlnet
del edges_controlnet
del seg_controlnet
if device == "mps":
    torch.mps.empty_cache()

# Step 2: temporal coherence between frames
coherence_pipeline = AnimateDiffVideoToVideoPipeline.from_pretrained(
    base_model,
    motion_adapter=adapter,
    torch_dtype=torch.float32,
).to(device)
coherence_pipeline.enable_attention_slicing()
coherence_pipeline.enable_vae_slicing()
coherence_pipeline.scheduler = DDIMScheduler.from_config(coherence_pipeline.scheduler.config)

guidance_scale_2 = 2.2
strength_2 = 0.22
num_inference_steps_2 = 10

inject_flow = [
    Image.blend(repainted[i].convert("RGB"), flow_parsed[i].resize(repainted[i].size).convert("RGB"), alpha=0.0)
    if i < len(flow_parsed) else repainted[i]
    for i in range(len(repainted))
]

def process_chunk(chunk):
    result = coherence_pipeline(
        video=chunk,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength_2,
        num_inference_steps=num_inference_steps_2,
        guidance_scale=guidance_scale_2,
        generator=generator,
    )
    return result.frames[0]

output_frames = []
jump = max(1, chunk_length - chunk_intersection)
start = 0
last_end = 0
while start < len(inject_flow):
    stop = min(start + chunk_length, len(inject_flow))
    c_frames = process_chunk(inject_flow[start:stop])
    shared_count = max(0, last_end - start)
    c_frames = c_frames[shared_count:]
    output_frames.extend(c_frames)
    last_end = stop
    if stop == len(inject_flow):
        break
    start += jump

final_frames = [frame.convert("RGB").resize(resolution) for frame in output_frames]
export_to_video(final_frames, output_path, fps=24, quality=9)
print("Output:", output_path)
