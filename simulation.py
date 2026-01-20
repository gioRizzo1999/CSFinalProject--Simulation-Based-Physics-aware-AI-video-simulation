import pybullet as p
import pybullet_data
import cv2
import numpy as np
import os
import json


p.connect(p.DIRECT)  # No GUI, faster for frame extraction
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# create output folder
os.makedirs("frames", exist_ok=True)

#------------------------------------- scene specs:
# what objects to load
# dimension, position, rotation
# camera position and rotation
# number of frames

# import json file with scene data
with open("parsed_json.json", "r") as f:
    scene = json.load(f)

# objects = scene.get("objects", []) 

# if plane in json list then load inclined plane
if "plane" in scene:
    plane_data = scene["plane"]
    pos = plane_data["position"]
    rot = plane_data["rotation"]
    plane_pos = [pos["x"], pos["y"], pos["z"]]
    plane_rot = [rot["x"], rot["y"], rot["z"]]

    print("plane detected")
    #load the plane
    plane_id = p.loadURDF("plane.urdf")
    # tilt the plane by rotating it around x-axis
    p.resetBasePositionAndOrientation(
        plane_id,
        plane_pos,
        p.getQuaternionFromEuler(plane_rot)   #(17 degrees inclination)
    )
else:
    print("No plane")


# if sphere in json list then load sphere
if "sphere" in scene:
    sphere_data = scene["sphere"]
    pos = sphere_data["position"]
    rot = sphere_data["rotation"]
    sphere_pos = [pos["x"], pos["y"], pos["z"]]
    sphere_rot = [rot["x"], rot["y"], rot["z"]]
    print("sphere detected")
    #load and position sphere
    sphere = p.loadURDF("sphere2.urdf", sphere_pos, p.getQuaternionFromEuler(sphere_rot))
else:
    print("No sphere")


# if cube in json list then load sphere
if "cube" in scene:
    cube_data = scene["cube"]
    pos = cube_data["position"]
    rot = cube_data["rotation"]
    cube_pos = [pos["x"], pos["y"], pos["z"]]
    cube_rot = [rot["x"], rot["y"], rot["z"]]
    print("cube detected")
    #load and position sphere
    cube = p.loadURDF("cube.urdf", cube_pos, p.getQuaternionFromEuler(cube_rot))
else:
    print("No cube")


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

# simulation loop

num_frames = 100  
#------------------------------------------
for i in range(num_frames):
    p.stepSimulation()

    # get camera image
    img = p.getCameraImage(width, height, view, proj)
    rgb = np.reshape(img[2], (height, width, 4))[:, :, :3]

    # save frame
    cv2.imwrite(f"frames/frame_{i:04d}.png", rgb)

p.disconnect()
