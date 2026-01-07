import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import os

depth_dir = "depth"
out_dir = "generated"
os.makedirs(out_dir, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using:", device)

# use full float32 for safety on MPS (slower but more stable)
dtype = torch.float32

# load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=dtype,
)

#load Stable Diffusion pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    feature_extractor=None,
    torch_dtype=dtype,
)

pipe.to(device)
pipe.enable_attention_slicing()

prompt = "a realistic ball rolling down an inclined plane, cinematic lighting, detailed textures"

# generate one image per depth map
for fname in sorted(os.listdir(depth_dir)):
    if not fname.endswith(".png"):
        continue

    depth_path = os.path.join(depth_dir, fname)
    out_path = os.path.join(out_dir, f"gen_{fname}")
    print("Generating:", fname)

    depth_img = Image.open(depth_path).convert("RGB")

    image = pipe(
        prompt=prompt,
        image=depth_img,
        num_inference_steps=25,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8, 
    ).images[0]

    image.save(out_path)

print("Done! Generated frames in:", out_dir)
