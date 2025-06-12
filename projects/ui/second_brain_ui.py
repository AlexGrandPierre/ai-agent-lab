import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess

AGENT_ROLES = {
    "Critic": """You are a critical thinking assistant. Your goal is to challenge the assumptions, logic, or implications of the idea or question, using relevant memory as context.""",
    "Explainer": """You are a philosophical explainer. Your goal is to clarify and break down complex ideas using the memory provided.""",
    "Synthesizer": """You are a synthesizer. Your goal is to combine ideas across different parts of memory to find underlying patterns or insights.""",
    "Designer": """You are a systems designer. Use the memory to propose actionable structures or models related to the idea.""",
    "Historian": """You are a contextual historian. Use memory to situate the idea within its intellectual or historical lineage.""",
}

# === Persistent Memory Setup ===
chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection = chroma_client.get_or_create_collection(name="second_brain")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Helper: Get Available Sources ===
def get_available_sources():
    try:
        metadatas = collection.get()["metadatas"]
        return sorted(set(meta.get("source", "unknown") for meta in metadatas if meta.get("source")))
    except Exception as e:
        print(f"Error fetching sources: {e}")
        return []

# === Retrieve Chunks by Query + Source ===
def retrieve_context(query, source_filter=None, top_k=10):
    query_embedding = embedder.encode([query]).tolist()[0]
    kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
    if source_filter:
        kwargs["where"] = {"source": source_filter}

    results = collection.query(**kwargs)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Diagnostic print
    print(f"\n🔍 Retrieved {len(docs)} chunks from source: {source_filter}")
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        print(f"Chunk {i+1}: {meta.get('source')} | {doc[:80]}...")

    return docs

# === Format + Query Ollama LLM ===
def ask_ollama_llm(query, context, model, role):
    role_instruction = AGENT_ROLES.get(role, "")
    prompt = f"""
{role_instruction}

### CONTEXT:
{context}

### QUESTION:
{query}

### INSTRUCTIONS:
- Be concise but insightful.
- Only use relevant context.
- Reflect the tone of your role.

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
        return f"Error querying model: {e}"

# === Main Agent Function ===
def second_brain_agent(query, model, source, role):
    chunks = retrieve_context(query, source_filter=source)
    context = "\n\n".join(chunks)
    response = ask_ollama_llm(query, context, model, role)
    return response, context


# === Gradio UI ===
with gr.Blocks(title="🧠 Second Brain Agent") as demo:
    gr.Markdown("## 🧠 Second Brain Agent with Document Reasoning\nAsk questions grounded in your own memory and selected documents.")

    with gr.Row():
        query_input = gr.Textbox(label="Ask a Question", placeholder="e.g., What does the AGI paper say about alignment?")
        model_choice = gr.Dropdown(["mistral", "llama2", "gemma:2b"], value="mistral", label="Choose Model")
        source_dropdown = gr.Dropdown(choices=get_available_sources(), label="Choose Document Source")
        agent_role = gr.Dropdown(choices=list(AGENT_ROLES.keys()), value="Explainer", label="Agent Role")

    output = gr.Textbox(label="Agent Response", lines=12)
    context_output = gr.Textbox(label="Retrieved Memory Context", lines=10)

    run_button = gr.Button("Ask the Agent")

    run_button.click(fn=second_brain_agent,
                 inputs=[query_input, model_choice, source_dropdown, agent_role],
                 outputs=[output, context_output])

demo.launch()
