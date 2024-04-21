from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

print(f"Cuda: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)

def encode_text(texts: List[str]):
    none_indexes = []
    for index in range(len(texts)):
        
        if type(texts[index]) != str:
            texts[index] = "0"
            none_indexes.append(index)

    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state[:,0,:].cpu().numpy().tolist()
        
    for index in none_indexes:
        embeddings[index] = [0.0] * model.config.hidden_size

    return embeddings