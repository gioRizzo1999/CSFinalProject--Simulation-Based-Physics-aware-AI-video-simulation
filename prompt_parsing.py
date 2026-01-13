#boilerplate code from Ollama Github + custom code
from ollama import chat
from ollama import ChatResponse

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

fineTuning = "turn the prompt into a json specification for a scene"

dynamicPrompt = "a realistic ball rolling down an inclined plane, cinematic lighting, detailed textures"

response: ChatResponse = chat(model='gemma:2b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)






