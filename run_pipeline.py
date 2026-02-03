#run pipeline files in correct order

#import all necessary files
#run fil1, then2, then

import subprocess
import sys

subprocess.run([sys.executable, "prompt_parsing.py"], check=True)
subprocess.run([sys.executable, "simulation.py"], check=True)
subprocess.run([sys.executable, "controlmap.py"], check=True)
subprocess.run([sys.executable, "generation_animateDiff.py"], check=True)