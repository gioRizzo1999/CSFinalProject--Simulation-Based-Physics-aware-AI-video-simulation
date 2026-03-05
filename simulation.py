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

objects = scene.get("objects", [])

OBJ_DICT = {
    "plane": "plane.urdf",
    "sphere": "sphere2.urdf",
    "cube": "cube.urdf",
}
# export objects with contrast in color to improve control maps strength
COLOR_BY_OBJECT = {
    "plane": [0.5, 0.5, 0.5, 1.0],   
    "cube":  [0.9, 0.5, 0.8, 0.1],   
    "sphere":[0.1, 0.2, 0.1, 0.6],   
}
found = {"plane": False, "sphere": False, "cube": False}
# loop over all objects and add to simulation if present
for o in objects:
    name = o.get("name")
    if name not in OBJ_DICT:
        continue
    if name == "plane" and found["plane"]:
        continue

    found[name] = True

    pos = o["position"]
    rot = o["rotation"]

    obj_pos = [pos["x"], pos["y"], pos["z"]]
    obj_rot = [rot["x"], rot["y"], rot["z"]]

    print(f"{name} detected")

    obj_id = p.loadURDF(OBJ_DICT[name])
    p.resetBasePositionAndOrientation(
        obj_id,
        obj_pos,
        p.getQuaternionFromEuler(obj_rot)
    )
    color = COLOR_BY_OBJECT.get(name, [0.7,0.7,0.7,1])
    try:
        p.changeVisualShape(obj_id, -1, rgbaColor=color, textureUniqueId=-1)
    except TypeError:
        p.changeVisualShape(obj_id, -1, rgbaColor=color)
# print absent objects
for k, v in found.items():
    if not v:
        print(f"No {k}")


# camera setup
width, height = 640, 480
view = p.computeViewMatrix(
    cameraEyePosition=[3, 0, 1],
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
