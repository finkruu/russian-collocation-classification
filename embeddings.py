import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

INPUT_FILE = "data/dataset_final_MULTICLASS.tsv"
OUTPUT_FILE = "data/dataset_with_embeddings.pkl"
MODEL_NAME = "cointegrated/rubert-tiny2"

def get_sentence_embedding(text, tokenizer, model):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    token_embeddings = model_output[0]
    attention_mask = encoded_input['attention_mask']
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return (sum_embeddings / sum_mask)[0].numpy()

def run():
    df = pd.read_csv(INPUT_FILE, sep='\t')
    df = df.dropna(subset=['Контекст']).reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    embeddings = []
    total_sentences = len(df)
    
    for i, text in enumerate(df['Контекст']):
        if (i + 1) % 50 == 0 or (i + 1) == total_sentences:
            print(f"Processing: {i + 1}/{total_sentences}")
            
        vec = get_sentence_embedding(text, tokenizer, model)
        embeddings.append(vec)
        
    df['Вектор'] = embeddings
    df.to_pickle(OUTPUT_FILE)

if __name__ == "__main__":
    run()