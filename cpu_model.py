import torch.neuron
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", torchscript=True)

example_input = "good"
# max_length = 128
# embeddings = tokenizer(example_input, max_length=max_length, padding="max_length",return_tensors="pt")
embeddings = tokenizer(example_input, return_tensors="pt")

save_dir = "cpu_model"
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, "model.pt"))
tokenizer.save_pretrained(save_dir)
model.config.save_pretrained(save_dir)
