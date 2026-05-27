from huggingface_hub import login
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import numpy as np
import os
import torch 

# Login to HuggingFace (via Token)
login(token=os.environ.get("HF_TOKEN"))

# Load PubMed biomedical abstracts
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:100]")
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

# Just to visualize how Llama breaks down the text into tokens (for the first abstract)
decoded_tokens = [tokenizer.decode([t]) for t in tokens["input_ids"][0]]
print("\nToken breakdown:")
for i, tok in enumerate(decoded_tokens):
    print(f"  [{i}] '{tok}'")
'''
'''
#EMBEDDINGS STAGE (generating vector representations of the abstracts for retrieval)
# Using Sentence traformers to generate embeddings for the abstracts (5 Abstracts). 

# Loading the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2") 

# Grab first 5 abstracts
texts = [dataset[i]["context"]["contexts"][0] for i in range(5)]

# Generate embeddings (hERE I'll visualize the shape and first few values of the embeddings)
embeddings = model.encode(texts)
print("Embedding shape:", embeddings.shape)
print("First embedding (values):", embeddings[0][:50])
'''
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
'''

# Testing similar embeddings from abstracts (Queries) with similarity
# Embed all abstracts
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding all abstracts...")
texts = [item["context"]["contexts"][0] for item in dataset]
questions = [item["question"] for item in dataset]
embeddings = model.encode(texts, show_progress_bar=True)

# Define a query topic
query = "role of mitochondria in programmed cell death"
query_embedding = model.encode([query])

# Find top 5 most similar abstracts to the query
scores = cosine_similarity(query_embedding, embeddings)[0]
top5_indices = np.argsort(scores)[::-1][:5]

print(f"\nTop 5 abstracts most similar to: '{query}'\n")

top_texts = []
top_questions = []

for rank, idx in enumerate(top5_indices):
    print(f"Rank {rank+1} — Score: {scores[idx]:.3f}")
    print(f"  Question: {questions[idx][:80]}...")
    top_texts.append(texts[idx])
    top_questions.append(questions[idx])

# Now  to run  the similarity matrix on these 5 related abstracts
top_embeddings = embeddings[top5_indices]
similarity_matrix = cosine_similarity(top_embeddings)

'''
print("\nSimilarity Matrix (top 5 related abstracts):")
print(np.round(similarity_matrix, 3))

print("\nPairwise Similarities:")
for i in range(5):
    for j in range(i+1, 5):
        score = similarity_matrix[i][j]
        print(f"\nAbstract {i+1} vs Abstract {j+1}: {score:.3f}")
        print(f"  Q{i+1}: {top_questions[i][:70]}...")
        print(f"  Q{j+1}: {top_questions[j][:70]}...")
'''

### INFERENCE STAGE (using Llama to answer a question based on the abstracts I rerieved)
## Adding inference to test the retrieval-augmented generation (RAG) pipeline with Llama
model_id = "microsoft/Phi-3-mini-4k-instruct"

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # float16 for GPU
    device_map="auto"           # auto-detects GPU on Niagara
)

# Use the top ranked abstract and its question
abstract = top_texts[0]
question = top_questions[0]

# Build prompt
prompt = f"""<|user|>
You are a biomedical research assistant.

Abstract:
{abstract}

Question: {question}

Answer yes or no and give a brief explanation.
<|end|>
<|assistant|>"""

print("\nRunning inference...")
inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

with torch.no_grad():
    outputs = llm.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        temperature=None,
        top_p=None
    )
# Decode only the new tokens (not the prompt)
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print("\n--- INPUT ---")
print(f"Question: {question}")
print(f"\nAbstract: {abstract[:200]}...")
print("\n--- MODEL OUTPUT ---")
print(response)