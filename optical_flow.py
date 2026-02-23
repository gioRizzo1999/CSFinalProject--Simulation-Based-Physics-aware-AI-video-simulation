# calculating dense optical flow using Gunnar Farnebäck's algorithm
import os
import cv2
import numpy as np

# define folders for input and output frames
depth_folder = "depth"
flow_folder = "flow"
os.makedirs(flow_folder, exist_ok=True)

# input
depth_frames = sorted([frame for frame in os.listdir(depth_folder) if frame.endswith(".png")])
resolution = (320, 240) 


# ensure that depth frames are greyscaled if not already
def grayscale_from_depth(frame):
    return cv2.normalize(frame.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

for i in range(len(depth_frames) - 1):
    dir1 = os.path.join(depth_folder, depth_frames[i])
    dir2 = os.path.join(depth_folder, depth_frames[i + 1])
    frame1 = grayscale_from_depth(cv2.imread(dir1, cv2.IMREAD_GRAYSCALE))
    frame2 = grayscale_from_depth(cv2.imread(dir2, cv2.IMREAD_GRAYSCALE))

    # resize depth to match generation resolution
    frame1norm = cv2.resize(frame1norm, resolution, interpolation=cv2.INTER_AREA)
    frame2norm = cv2.resize(frame2norm, resolution, interpolation=cv2.INTER_AREA)

    flow_frames = cv2.calcOpticalFlowFarneback(frame1norm, frame2norm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    np.save(os.path.join(flow_folder, f"flow_{i:04d}.npy"), flow_frames)

print("Optical flow frames generated.")