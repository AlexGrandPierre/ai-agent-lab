import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess

# === Setup ===
COLLECTION_NAME = "second_brain"
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# === Retrieve memory chunks ===
def retrieve_context(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

# === LLM reasoning via Ollama ===
def ask_ollama_llm(query, context, model):
    prompt = f"""
You are a reasoning assistant. Use the CONTEXT below to answer the QUESTION that follows.

### CONTEXT:
{context}

### QUESTION:
{query}

### ANSWER:
"""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Error: {e}"

# === Gradio logic ===
def second_brain_agent(query, model):
    context_chunks = retrieve_context(query)
    context = "\n\n".join(context_chunks)
    answer = ask_ollama_llm(query, context, model)
    return answer, context

# === Gradio UI ===
with gr.Blocks(title="🧠 Second Brain Agent") as demo:
    gr.Markdown("## 🧠 Second Brain Agent\nAsk questions grounded in your own memory + notes.")

    with gr.Row():
        query_input = gr.Textbox(label="Ask a question", placeholder="e.g., What did I say about epistemic neutrality?")
        model_choice = gr.Dropdown(["llama2", "mistral", "gemma:2b"], value="mistral", label="Model")

    output = gr.Textbox(label="Agent Response", lines=12)
    context_output = gr.Textbox(label="Retrieved Memory Context", lines=10)

    run_button = gr.Button("Ask the Agent")
    run_button.click(fn=second_brain_agent, inputs=[query_input, model_choice], outputs=[output, context_output])

demo.launch()
