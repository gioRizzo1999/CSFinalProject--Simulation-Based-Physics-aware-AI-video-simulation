#run pipeline files in correct order

#import all necessary files
#run fil1, then2, ecc

import subprocess
import sys

def run_pipeline(user_prompt):
    subprocess.run([sys.executable, "prompt_parsing.py", user_prompt], check=True)
    subprocess.run([sys.executable, "simulation.py"], check=True)
    subprocess.run([sys.executable, "controlmap.py"], check=True)
    subprocess.run([sys.executable, "generation_animateDiff.py"], check=True)

