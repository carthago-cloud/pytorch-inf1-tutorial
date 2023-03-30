import torch.neuron
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", torchscript=True)

example_input = "good"
# max_length = 128
# embeddings = tokenizer(example_input, max_length=max_length, padding="max_length",return_tensors="pt")
embeddings = tokenizer(example_input, return_tensors="pt")
neuron_inputs = tuple(embeddings.values())

model_neuron = torch.neuron.trace(model, neuron_inputs)


save_dir = "tmp"
os.makedirs("tmp", exist_ok=True)
model_neuron.save(os.path.join(save_dir, "neuron_model.pt"))
tokenizer.save_pretrained(save_dir)
model.config.save_pretrained(save_dir)

#     # run prediciton
#     with torch.no_grad():
#         predictions = model(*neuron_inputs)[0]
#         scores = torch.nn.Softmax(dim=1)(predictions)
#
#     # return dictonary, which will be json serializable
#     return [{"label": model_config.id2label[item.argmax().item()], "score": item.max().item()} for item in scores]
#
# output = model(encoded_input)
# print(output)
