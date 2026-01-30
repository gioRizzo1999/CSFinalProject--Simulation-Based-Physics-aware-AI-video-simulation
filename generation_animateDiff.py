# animate diffusion on controlnet frames

import os
import torch
from PIL import Image

from diffusers import (
    AnimateDiffControlNetPipeline,
    ControlNetModel,
    MotionAdapter,
    DDIMScheduler,
)
from diffusers.utils import export_to_gif

#setup for mps
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32  

