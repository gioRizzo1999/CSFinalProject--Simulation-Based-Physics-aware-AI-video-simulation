#boilerplate code from Ollama Github + custom code
from ollama import chat
from ollama import ChatResponse
import json

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

response: ChatResponse = chat(model='gemma:2b', messages=[
  {
    'role': 'user',
    'content': "begginning of scene: a realistic cube rolling down an inclined plane, cinematic lighting, detailed textures. end of scene. Determine which objects are mentioned in the scene and output them as a json list of strings in this format:[, , ], do not add anything else",
  },
])

# print(response['message']['content'])
# or access fields directly from the response object
# print(response.message.content)


#parse into json
string = response.message.content
parsed_json = json.loads(string)
print(parsed_json)

#select only content between curly brackets
# input_string = string
# isolated_string = input_string.split('{')[1].split('}')[0]
# print(isolated_string)  

# write to file
with open("parsed_json.json", "w") as file:
  json.dump(parsed_json, file, indent=4)

#create tests to validate json parsing