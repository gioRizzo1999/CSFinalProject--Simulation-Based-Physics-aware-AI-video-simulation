import torch
import os
from diffusers import AnimateDiffControlNetPipeline, ControlNetModel, MotionAdapter
from diffusers.utils import export_to_gif
from PIL import Image


# setting backend
device = "mps" if torch.backends.mps.is_available() else "cpu"

# folders
input_maps = "depth"
output_path = "output.gif"
resolution = (224,192)
frames_num = 4

# load conditioning frames
frames = sorted(os.listdir(input_maps))[:frames_num]
input_frames = [
    Image.open(os.path.join(input_maps, f)).convert("RGB").resize(resolution)
    for f in frames
]

# models
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth"
)

adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2"
)

gen_pipeline = AnimateDiffControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    motion_adapter=adapter,
).to(device)


# prompts
prompt = "a red cube falling on a blue inclined plane, minimal background"
negative_prompt = "blurry, artifacts"

# diffusion knobs
guidance_scale = 7.0
controlnet_conditioning_scale = 1.2
num_inference_steps = 22

# fixed seed for deterministic outputs
generator = torch.Generator(device="cpu").manual_seed(42)

# generation function
with torch.inference_mode():
    result = gen_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        conditioning_frames=input_frames,
        num_frames=len(input_frames),
        generator=generator,
        guidance_scale = guidance_scale,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        num_inference_steps = num_inference_steps,
    )
output_frames = result.frames[0]


# save to file
export_to_gif(output_frames, output_path, fps=8)
print("Output:", output_path)