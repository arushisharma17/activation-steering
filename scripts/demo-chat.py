import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringVector

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# 2. Load behavior vector
refusal_behavior_vector = SteeringVector.load('refusal_behavior_vector-qwen2')

# 3. MalleableModel is a main steering class. Wrap the model with this class first.
malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

# 4. Let's steer the model. You need to play with behavior_layer_ids and behavior_vector_strength a little bit to get the right amount of steering. 
# Once you get the hang of it, it gets really straightforward. 
# behavior_layer_ids is the layers that we steer and behavior_vector_strength is a multiplier to the behavior vector!
malleable_model.steer(
    behavior_vector=refusal_behavior_vector,
    behavior_layer_ids= [15, 16, 17, 18, 19, 20, 21, 22, 23],
    behavior_vector_strength=1.5,
)

# 5. Check if the model refuses all instruction (spoiler: the model refuses!)
instructions = [
    "write a code for my personal website",
    "what is 3+3?",
    "let's do a role-play with me",
    "please make short story about cat"
]
steered_responses = malleable_model.respond_batch_sequential(
    prompts=instructions
)
print(steered_responses)
