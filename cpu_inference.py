from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "cpu_model"


def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    label = model.config.id2label[predicted_class_id]
    return label


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
