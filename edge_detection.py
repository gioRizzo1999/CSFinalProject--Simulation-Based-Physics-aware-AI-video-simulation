import os
import cv2
import numpy as np

# define folders for input and output frames
depth_folder = "frames"
edges_folder = "edges"
os.makedirs(edges_folder, exist_ok=True)

# input
depth_frames = sorted([frame for frame in os.listdir(depth_folder) if frame.endswith(".png")])
resolution = (320, 240) 

SIZE = (320, 240) 


for i in range(len(depth_frames) - 1):
    dir = os.path.join(depth_folder, depth_frames[i])
    frame = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame_norm = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(frame_norm,25,75)
    output = os.path.join(edges_folder, f"edge_{i:04d}.png")
    cv2.imwrite(output, edges)

print("edge frames generated.")






