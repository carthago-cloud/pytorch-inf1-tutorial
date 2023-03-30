import os
from transformers import AutoConfig, AutoTokenizer
import torch

MODEL_DIR = "neuron_model.py"

def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    label = model_config.id2label[predicted_class_id]
    return label


# saved weights name
WEIGHTS_FILE_NAME = "model.pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = torch.jit.load(os.path.join(MODEL_DIR, WEIGHTS_FILE_NAME))
model_config = AutoConfig.from_pretrained(MODEL_DIR)

print(predict("bad"))
print(predict("good"))
