import chromadb
from sentence_transformers import SentenceTransformer
import subprocess
import os

# === Setup ===
COLLECTION_NAME = "second_brain"
DATA_FILE = "data/thinking_partner_log.txt"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"  # Change to llama2, gemma, etc.

embedder = SentenceTransformer(EMBED_MODEL)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# === Load and Embed Text ===
def load_and_store_text(file_path, chunk_size=500):
    with open(file_path, "r") as f:
        text = f.read()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    embeddings = embedder.encode(chunks).tolist()
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"✅ Loaded and embedded {len(chunks)} chunks.")

# === Retrieve Relevant Context ===
def retrieve_context(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

# === Send to LLM ===
def ask_ollama_llm(query, context, model=OLLAMA_MODEL):
    formatted_prompt = f"""
You are a context-aware reasoning assistant. Use the following context to answer the user's question thoughtfully.

### CONTEXT:
{context}

### QUESTION:
{query}

### ANSWER:
"""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=formatted_prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Error: {e}"

# === Main Loop ===
if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found.")
    else:
        print(f"🔍 Loading: {DATA_FILE}")
        load_and_store_text(DATA_FILE)

        print(f"\n🤖 Second Brain RAG Agent Ready (Model: {OLLAMA_MODEL})")
        print("Type 'exit' to quit.\n")

        while True:
            query = input("Ask a question: ")
            if query.lower() == "exit":
                break
            context_docs = retrieve_context(query)
            context = "\n\n".join(context_docs)
            response = ask_ollama_llm(query, context)
            print(f"\n🧠 Agent Response:\n{response}\n{'-'*50}")
