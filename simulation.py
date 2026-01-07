import pybullet as p
import pybullet_data
import cv2
import numpy as np
import os


p.connect(p.DIRECT)  # No GUI, faster for frame extraction
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# create output folder
os.makedirs("frames", exist_ok=True)

#------------------------------------- scene specs
# load inclined plane
plane_id = p.loadURDF("plane.urdf")

# tilt the plane by rotating it around x-axis
p.resetBasePositionAndOrientation(
    plane_id,
    [0, 0, 0],
    p.getQuaternionFromEuler([0.3, 0, 0])   #(17 degrees inclination)
)

# load sphere
ball_start_pos = [0, 0, 0.5]
ball_start_orn = p.getQuaternionFromEuler([0, 0, 0])
ball = p.loadURDF("sphere2.urdf", ball_start_pos, ball_start_orn)

# camera setup
width, height = 640, 480
view = p.computeViewMatrix(
    cameraEyePosition=[1.5, 0, 1],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 0, 1],
)
proj = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=width / height,
    nearVal=0.1,
    farVal=10.0
)
#------------------------------------------

# simulation loop
num_frames = 100  
for i in range(num_frames):
    p.stepSimulation()

    # get camera image
    img = p.getCameraImage(width, height, view, proj)
    rgb = np.reshape(img[2], (height, width, 4))[:, :, :3]

    # save frame
    cv2.imwrite(f"frames/frame_{i:04d}.png", rgb)

p.disconnect()
