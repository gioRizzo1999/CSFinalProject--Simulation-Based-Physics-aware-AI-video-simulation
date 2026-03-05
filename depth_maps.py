# script inspired by MiDaS documentation: https://pytorch.org/hub/intelisl_midas_v2/
import torch
import os
import cv2
import numpy as np

# load mps backend for running Ai model on Apple M3 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# define folders for input and output frames
frames_folder = "frames"
output_folder = "depth"
os.makedirs(output_folder, exist_ok=True)


# load MiDaS model
#model_type = "DPT_Large"     # MiDaS v3 - Large  
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid   
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small  
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True).to(device)
midas.eval()

model_tranforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = model_tranforms.dpt_transform

# processing all frames 
for frame in sorted(os.listdir(frames_folder)):
    if not frame.endswith(".png"):
        continue

    input_path = os.path.join(frames_folder, frame)
    output_path = os.path.join(output_folder, f"depth_{frame}")
    img = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_greyscaled = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(output_path, depth_greyscaled)

print("Successully created depth maps in:", output_folder)
