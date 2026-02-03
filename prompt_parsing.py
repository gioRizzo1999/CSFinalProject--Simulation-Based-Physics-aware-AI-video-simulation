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

user_prompt = sys.argv[1]
# user_prompt = "a realistic sphere falling from above over an inclined plane 30 degrees, minimal scene, realistic layout and motion."


tuning= """ Based on the description of the scene, produce a JSON file that matches the following schema, Use the same keys, Replace example values with values you deem appropriate to the scene: {
  "objects": [
    {
      "name": "string, choose between plane, sphere, cube",
      "how_many": int,
      "position": { "x": radians, "y": radians, "z": radians },
      "rotation": { "x": radians, "y": radians, "z": radians }
    },
    {
      "name": "string, choose between plane, sphere, cube",
      "how_many": int,
      "position": { "x": radians, "y": radians, "z": radians },
      "rotation": { "x": radians, "y": radians, "z": radians }
    }
  ]
}
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

if start == -1 or end == 0:
    raise ValueError(f"No JSON found: \n{resp}")

only_json = resp[start:end]
parsed_json = json.loads(only_json)


#select only content between curly brackets
# input_string = string
# isolated_string = input_string.split('{')[1].split('}')[0]
# print(isolated_string)  

# write to file
with open("parsed_json.json", "w") as file:
  json.dump(parsed_json, file, indent=4)

#create tests to validate json parsing


