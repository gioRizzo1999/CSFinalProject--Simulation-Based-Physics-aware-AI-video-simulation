#boilerplate code from Ollama Github + custom code
from ollama import chat
from ollama import ChatResponse
import json
import sys

#high level:
#user inserts prompt that describes scene
#llm is finetuned to interpret every prompt as a scene description
#llm extracts useful descriptive data from the prompt
#outputs data or code for pybullet simulation file

#technically:
# -decide finetuning prompt structure: 
# ----what does pybullet wants minimally? json? code? what initial bounds to put in place? 
# ----create test with various inputs 
# -apply it to llm request code
# -decide how to validate llm output
# -validate llm output
# -pass output to simulation.py

#minimal: produce json spec
#final idea: produce json specs and intent, then trigger pre-defined code for more complex scene

# fineTuning = "turn the prompt into a json specification for a scene. " \
# "Extract relevant data in the prompt and put it in the following format:" \
# ""

# user_prompt = sys.argv[1]
user_prompt = "a red cube made in plastic falling from above over an inclined plane 30 degrees, minimal scene, realistic layout and motion."


# tuning= """ Based on the description of the scene, produce a JSON file that matches the following schema, Use the same keys, Replace example values with values you deem appropriate to the scene: {
#   "objects": [
#     {
#       "name": "string, choose between plane, sphere, cube",
#       "how_many": int,
#       "position": { "x": radians, "y": radians, "z": radians },
#       "rotation": { "x": radians, "y": radians, "z": radians }
#     },
#     {
#       "name": "string, choose between plane, sphere, cube",
#       "how_many": int,
#       "position": { "x": radians, "y": radians, "z": radians },
#       "rotation": { "x": radians, "y": radians, "z": radians }
#     }
#   ]
# }
# """

tuning= """ Based on the scene description, output ONLY a valid JSON object matching exactly this schema (same keys). Do not add any text before/after the JSON and do not use markdown.
Units: position is in meters (x,y in [-2,2], z in [0,3]); rotation is in radians (x,y,z in [-3.14,3.14]).
Constraints: name must be one of ["plane","sphere","cube"]; how_many must be an integer in [1,3].
Schema: { "objects":[ { "name":"plane|sphere|cube", "how_many": 1, "position":{"x":0.0,"y":0.0,"z":0.0}, "rotation":{"x":0.0,"y":0.0,"z":0.0} } ] }
"""

full_prompt = f"""{user_prompt} {tuning}""" 

response: ChatResponse = chat(model='qwen3-coder:480b-cloud', messages=[
  {
    'role': 'user',
    'content': full_prompt
  },
])

# print(response['message']['content'])
# or access fields directly from the response object
# print(response.message.content)


#parse into json
# string = response.message.content
# parsed_json = json.loads(string)
# print(parsed_json)

resp = response.message.content.strip()
start = resp.find("{")
end = resp.rfind("}") + 1

# format checks
if start == -1 or end <= start:
    raise ValueError(f"No JSON found: \n{resp}")

only_json = resp[start:end]

try:
    parsed_json = json.loads(only_json)
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON:\n{only_json}")

# write to file
with open("parsed_json.json", "w") as file:
  json.dump(parsed_json, file, indent=4)

#create tests to validate json parsing


