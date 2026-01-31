import torch
import os
from diffusers import AnimateDiffControlNetPipeline, ControlNetModel, MotionAdapter
from diffusers.utils import export_to_gif
from PIL import Image


# setting backend
device = "mps" if torch.backends.mps.is_available() else "cpu"

# folders
input_maps = "depth"
output_path = "ouput.gif"
resolution = (256, 240)
frames_num = 8

# load conditioning frames
frames = sorted(os.listdir(input_maps))[:frames_num]
input_frames = [
    Image.open(os.path.join(input_maps, f)).convert("RGB").resize(resolution)
    for f in frames
]