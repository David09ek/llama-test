from huggingface_hub import login
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import numpy as np
import os

# Login to HuggingFace (via Token)
login(token=os.environ.get("HF_TOKEN"))

# Loading a small public dataset to test Hugging Face Integration
#dataset = load_dataset("ag_news", split="train[:5]")

# Load PubMed biomedical abstracts
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:100]")

# I'll print a row ot ocnformt at evreything works perfectly;
#print(dataset[0])

'''
# Load Llama's tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Take the first abstract
sample_text = dataset[0]["context"]["contexts"][0]
print("Original Text:")
print(sample_text)
print(f"\nWord count: {len(sample_text.split())}")

# Tokenize it
tokens = tokenizer(sample_text, return_tensors="pt")
print(f"\nToken count: {tokens['input_ids'].shape[1]}")

# See exactly how Llama breaks the text down
decoded_tokens = [tokenizer.decode([t]) for t in tokens["input_ids"][0]]
print("\nToken breakdown:")
for i, tok in enumerate(decoded_tokens):
    print(f"  [{i}] '{tok}'")
'''

#EMBEDDINGS STAGE (generating vector representations of the abstracts for retrieval)
# Using Sentence traformers to generate embeddings for the abstracts (5 Abstracts). 

# Loading the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2") 

# Grab first 5 abstracts
texts = [dataset[i]["context"]["contexts"][0] for i in range(5)]

# Generate embeddings (hERE I'll visualize the shape and first few values of the embeddings)
embeddings = model.encode(texts)
'''
print("Embedding shape:", embeddings.shape)
print("First embedding (values):", embeddings[0][:50])
'''

# Compute similarity between all 5 abstracts
similarity_matrix = cosine_similarity(embeddings)

print("\nSimilarity Matrix (5x5):")
print(np.round(similarity_matrix, 3))

# Make it more readable — print each pair with their questions
print("\nPairwise Similarities:")
for i in range(5):
    for j in range(i+1, 5):
        score = similarity_matrix[i][j]
        q_i = dataset[i]["question"][:60]
        q_j = dataset[j]["question"][:60]
        print(f"\nAbstract {i+1} vs Abstract {j+1}: {score:.3f}")
        print(f"  Q{i+1}: {q_i}...")
        print(f"  Q{j+1}: {q_j}...")