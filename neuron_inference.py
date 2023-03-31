import os
from transformers import AutoConfig, AutoTokenizer
import torch
import torch.neuron

MODEL_DIR = "neuron_model"


def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    neuron_inputs = tuple(inputs.values())
    with torch.no_grad():
        logits = model(*neuron_inputs)[0]
    predicted_class_id = logits.argmax().item()
    label = model_config.id2label[predicted_class_id]
    return label


# To use one neuron core per worker
os.environ["NEURON_RT_NUM_CORES"] = "1"

# saved weights name
AWS_NEURON_TRACED_WEIGHTS_NAME = "model.pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = torch.jit.load(os.path.join(MODEL_DIR, AWS_NEURON_TRACED_WEIGHTS_NAME))
model_config = AutoConfig.from_pretrained(MODEL_DIR)
