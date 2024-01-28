# This is a sample Python script.
import torch
import torch.nn.functional as F
from reader import embed_pdf

from transformers import DistilBertTokenizer, DistilBertModel, pipeline
from transformers.modeling_outputs import BaseModelOutput

pdf_data = [
    "Shantanu is a man with 5 hands and strength of 2 elephants, standing 9 feet tall and can fly",
    "Can jump distances of 10 metres, and run with a speed to 30km/h",
    "Can swim upto a speed of 15km/h",
    "he lived in Madhya Pradesh"
]

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

encoded_pdf_data = tokenizer(pdf_data, return_tensors='pt', padding=True)

outputs = model(**encoded_pdf_data)
embeddings_data = outputs.last_hidden_state
print(f"pdf data embedding shape:",embeddings_data.shape)

question = "swim speed is"
question_input = tokenizer(question, return_tensors='pt')
encoded_input = model(**question_input)
encoded_input = encoded_input.last_hidden_state
print("question embedding input shape",encoded_input.shape)

question_embedding = encoded_input.mean(dim=1)
print("question embedding converted shape",question_embedding.shape)
pdf_data_embeddings = embeddings_data.mean(dim=1)
print("pdf data embedding converted shape",pdf_data_embeddings.shape)

similarities = [F.cosine_similarity(question_embedding, pdf_data_embedding, dim=1) for pdf_data_embedding in
                pdf_data_embeddings]

print(similarities)

max_index = torch.argmax(torch.tensor(similarities))

print(max_index)

print("Matching input from pdf data is :", pdf_data[max_index])


print(embed_pdf("data/demo.pdf"))