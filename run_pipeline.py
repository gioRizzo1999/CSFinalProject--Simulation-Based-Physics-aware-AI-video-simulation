#run pipeline files in correct order

import subprocess
import sys

def run_pipeline(user_prompt):
    subprocess.run([sys.executable, "prompt_parsing.py", user_prompt], check=True)
    subprocess.run([sys.executable, "simulation.py"], check=True)
    subprocess.run([sys.executable, "depth_maps.py"], check=True)
    subprocess.run([sys.executable, "optical_flow.py"], check=True)
    subprocess.run([sys.executable, "edge_detection.py"], check=True)
    subprocess.run([sys.executable, "video_gen_long.py", user_prompt], check=True)

    return "/Users/gioriz/Desktop/final project/project_files/output_long_test6.mp4"


# uncomment below to test script from console/without frontend
# prompt = "a blue soccer football sliding down a green inclined plane, minimal scene, fixed camera, sharp details"
# run_pipeline(prompt)
