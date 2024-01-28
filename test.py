import string

from transformers import DistilBertTokenizer, DistilBertModel, pipeline
from transformers.modeling_outputs import BaseModelOutput
import torch
from reader import read_txt_file


def generate_file_embeddings(file_data: list[string]):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    encoded_file_data = tokenizer(file_data, return_tensors='pt', padding=True)
    outputs = model(**encoded_file_data)
    embeddings_data: torch.FloatTensor = outputs.last_hidden_state
    print(f"pdf data embedding shape:", embeddings_data.shape)
    return embeddings_data


def generate_question_embedding(question: string):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    question_input = tokenizer(question, return_tensors='pt')
    encoded_input = model(**question_input)
    question_embedding: torch.FloatTensor = encoded_input.last_hidden_state
    return question_embedding


def perform_similarity_check(question_embedding: torch.FloatTensor, file_embeddings: torch.FloatTensor):
    mean_file_embeddings = file_embeddings.mean(dim=1)
    mean_question_embedding = question_embedding.mean(dim=1)
    print("question embedding converted shape", question_embedding.shape)
    similarities = [torch.cosine_similarity(mean_question_embedding, mean_file_embedding, dim=1) for mean_file_embedding in
                    mean_file_embeddings]
    print(similarities)
    # get index of similarity which is >0.3
    filtered_similarities = [index for index, m in enumerate(similarities) if m > 0.3]
    return filtered_similarities

def main():
    file_path = "data/demo.txt"
    question = "tax evasion"
    file_data = read_txt_file(file_path)
    file_embeddings = generate_file_embeddings(file_data)
    question_embedding = generate_question_embedding(question)
    matched_input = perform_similarity_check(question_embedding, file_embeddings)
    print("Matching input from file data is :")
    for index in matched_input:
        print(file_data[index])

main()