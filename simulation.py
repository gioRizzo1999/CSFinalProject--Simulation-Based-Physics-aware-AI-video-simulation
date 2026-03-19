import pybullet as p
import pybullet_data
import cv2
import numpy as np
import os
import json


p.connect(p.DIRECT)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
physics_steps = 60
ai_video_fps = 24
steps_per_frame = max(1, int(physics_steps / ai_video_fps))
p.setTimeStep(1.0 / physics_steps)

os.system("rm -rf frames segmentation flow edges depth && mkdir -p frames segmentation flow edges depth")

#------------------------------------- scene specs:
# what objects to load
# dimension, position, rotation
# camera position and rotation
# number of frames

# camera setup
width, height = 512, 384
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
objects_ordered = sorted(objects, key=lambda o: 0 if o.get("name") == "plane" else 1)
for o in objects_ordered:
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
    if name != "plane":
        p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])
        p.changeDynamics(obj_id, -1, linearDamping=0.04, angularDamping=0.04, restitution=0.4)
    else:
        p.changeDynamics(obj_id, -1, restitution=0.4)
    color = COLOR_BY_OBJECT.get(name, [0.7,0.7,0.7,1])
    try:
        p.changeVisualShape(obj_id, -1, rgbaColor=color, textureUniqueId=-1)
    except TypeError:
        p.changeVisualShape(obj_id, -1, rgbaColor=color)
    if name == "plane":
        os.makedirs("style_anchor", exist_ok=True)
        rgb = np.reshape(p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)[2], (height, width, 4))[:, :, :3]
        cv2.imwrite("style_anchor/anchor_image.png", rgb)


# print absent objects
for k, v in found.items():
    if not v:
        print(f"no {k}")


# simulation loop

num_frames = 50
#------------------------------------------
for i in range(num_frames):
    for _ in range(steps_per_frame):
        p.stepSimulation()

    # get camera image
    img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(img[2], (height, width, 4))[:, :, :3]
    seg_maps = np.reshape(img[4], (height, width))
    object_ids = seg_maps % (1 << 24)
    seg_img = cv2.normalize(object_ids.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    seg_img = seg_img.astype(np.uint8)

    # save frames and maps
    cv2.imwrite(f"frames/frame_{i:04d}.png", rgb)
    cv2.imwrite(f"segmentation/seg_{i:04d}.png", seg_img)
    np.save(f"segmentation/seg_{i:04d}.npy", object_ids)

p.disconnect()
