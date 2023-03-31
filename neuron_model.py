import torch.neuron
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", torchscript=True)

example_input = "good"
embeddings = tokenizer(example_input, return_tensors="pt")
neuron_inputs = tuple(embeddings.values())

model_neuron = torch.neuron.trace(model, neuron_inputs)


save_dir = "neuron_model"
os.makedirs("neuron_model", exist_ok=True)
model_neuron.save(os.path.join(save_dir, "model.pt"))
tokenizer.save_pretrained(save_dir)
model.config.save_pretrained(save_dir)
