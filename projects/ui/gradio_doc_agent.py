import gradio as gr
import PyPDF2
import os
import subprocess
import datetime
import re

LOG_FILE = "summary_log.txt"
MODEL_OPTIONS = ["llama2", "gemma", "mistral", "deepseek-coder:1.3b"]

def extract_text(file_path):
    try:
        if isinstance(file_path, list):
            file_path = file_path[0]
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
        else:
            return "Unsupported file type."
        return text.encode("utf-8", errors="replace").decode("utf-8")
    except Exception as e:
        return f"Extraction error: {str(e)}"

def summarize_with_model(model, text):
    prompt = (
        "Summarize the following document in bullet points. "
        "Then provide a two-sentence analysis:\n\n"
        + text[:3000]
    )
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        response = result.stdout.decode("utf-8").strip()
        response = response.encode("utf-8", errors="replace").decode("utf-8")
        log_summary_interaction(model, prompt, response)
        return response
    except Exception as e:
        return f"Summarization error: {str(e)}"

def log_summary_interaction(model, prompt, response):
    timestamp = datetime.datetime.now().isoformat()
    log_entry = (
        f"\n[{timestamp}]\n"
        f"Model: {model}\n"
        f"Prompt (Truncated):\n{prompt[:500]}...\n"
        f"Response:\n{response}\n"
        f"{'-'*60}\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

def load_logs(log_path="summary_log.txt"):
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        entries = content.strip().split("\n" + "-" * 60 + "\n")
        return entries[::-1]  # Most recent first
    except FileNotFoundError:
        return []

def search_logs(query, log_path="summary_log.txt"):
    entries = load_logs(log_path)
    results = [entry for entry in entries if query.lower() in entry.lower()]
    if not results:
        return "No matches found."
    return "\n\n".join(results[:5])  # Show up to 5 matches

def export_logs_to_markdown(log_path="summary_log.txt", output_folder="markdown_logs"):
    os.makedirs(output_folder, exist_ok=True)
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return "No log file found."

    entries = content.strip().split("\n" + "-" * 60 + "\n")
    exported_files = []

    for idx, entry in enumerate(entries):
        # Extract timestamp for filename
        match = re.search(r"\[(.*?)\]", entry)
        timestamp = match.group(1).replace(":", "-") if match else f"entry_{idx}"
        filename = f"{timestamp}.md"
        filepath = os.path.join(output_folder, filename)

        # Format markdown
        markdown_entry = entry.replace("Model:", "## Model:")\
                              .replace("Prompt (Truncated):", "### Prompt (Truncated)")\
                              .replace("Response:", "### Response")

        with open(filepath, "w", encoding="utf-8") as out:
            out.write(f"# Summary Log Entry\n\n{markdown_entry.strip()}")
        exported_files.append(filepath)

    return f"Exported {len(exported_files)} logs to '{output_folder}'"

def export_logs_to_markdown(log_path="summary_log.txt", output_folder="markdown_logs"):
    os.makedirs(output_folder, exist_ok=True)
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return "No log file found."

    entries = content.strip().split("\n" + "-" * 60 + "\n")
    exported_files = []

    for idx, entry in enumerate(entries):
        # Extract timestamp
        match = re.search(r"\[(.*?)\]", entry)
        timestamp = match.group(1).replace(":", "-") if match else f"entry_{idx}"
        filename = f"{timestamp}.md"
        filepath = os.path.join(output_folder, filename)

        # Convert to markdown
        markdown_entry = entry.replace("Model:", "## Model:")\
                              .replace("Prompt (Truncated):", "### Prompt (Truncated)")\
                              .replace("Response:", "### Response")

        with open(filepath, "w", encoding="utf-8") as out:
            out.write(f"# Summary Log Entry\n\n{markdown_entry.strip()}")

        exported_files.append(filepath)

    return f"Exported {len(exported_files)} logs to '{output_folder}'"
    
with gr.Blocks() as demo:
    gr.Markdown("### Upload, Extract, and Summarize a Document")

    file_input = gr.File(label="Upload", file_types=[".txt", ".pdf"])
    extract_button = gr.Button("Extract Text")
    document_display = gr.Textbox(lines=15, label="Extracted Text")

    model_selector = gr.Dropdown(choices=MODEL_OPTIONS, value="llama2", label="Choose Model")
    summarize_button = gr.Button("Summarize")
    summary_output = gr.Textbox(lines=10, label="Summary")

    extract_button.click(fn=extract_text, inputs=file_input, outputs=document_display)
    summarize_button.click(fn=summarize_with_model, inputs=[model_selector, document_display], outputs=summary_output)

    with gr.Accordion("🔎 Search Past Summaries", open=False):
        search_input = gr.Textbox(label="Search Query")
        search_button = gr.Button("Search Logs")
        search_output = gr.Textbox(label="Search Results", lines=12)

        search_button.click(fn=search_logs, inputs=search_input, outputs=search_output)

    with gr.Accordion("📤 Export Logs to Markdown", open=False):
        export_button = gr.Button("Export Logs")
        export_status = gr.Textbox(label="Export Status")

        export_button.click(fn=export_logs_to_markdown, inputs=[], outputs=export_status)


demo.launch()
