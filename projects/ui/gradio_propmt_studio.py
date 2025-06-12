import gradio as gr
import subprocess
import datetime
import os

MODEL_OPTIONS = ["llama2", "mistral", "gemma:2b", "deepseek-coder:1.3b"]

TASK_TEMPLATES = {
    "Freeform Prompt": "",
    "Refine Idea/Thesis": "Refine this concept for clarity, logic, and insight: ",
    "Generate App Skeleton": "Generate the basic code for a web app that does the following: ",
    "Create Visual Art Prompt": "Create a detailed prompt for an AI art generator to produce: ",
    "Simulate Multi-Agent Discussion": "Simulate a conversation between 3 AI agents with distinct goals about: ",
    "Summarize or Index Notes": "Summarize and extract key concepts from the following notes: "
}

LOG_FILE = "agent_task_log.txt"

def query_ollama_gradio(model, prompt, task):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        response = result.stdout.decode("utf-8").strip()
    except Exception as e:
        response = f"Error: {e}"

    log_interaction(model, task, prompt, response)
    return response

def log_interaction(model, task, prompt, response):
    timestamp = datetime.datetime.now().isoformat()
    log_entry = (
        f"\n[{timestamp}]\n"
        f"Task: {task}\n"
        f"Model: {model}\n"
        f"Prompt:\n{prompt}\n"
        f"Response:\n{response}\n"
        f"{'-'*60}\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

def set_prompt_from_task(task):
    return TASK_TEMPLATES.get(task, "")

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Local AI Agent Studio")
    gr.Markdown("Select a task, choose a model, and interact with AI agents. All outputs are logged.")

    with gr.Row():
        model_selector = gr.Dropdown(choices=MODEL_OPTIONS, value="llama2", label="Model")
        task_selector = gr.Dropdown(choices=list(TASK_TEMPLATES.keys()), value="Freeform Prompt", label="Task")

    prompt_input = gr.Textbox(lines=4, placeholder="Enter your prompt here...", label="Prompt")
    task_selector.change(fn=set_prompt_from_task, inputs=task_selector, outputs=prompt_input)

    run_button = gr.Button("Run Agent")
    output_box = gr.Textbox(label="Agent Response", lines=6)

    run_button.click(fn=query_ollama_gradio, inputs=[model_selector, prompt_input, task_selector], outputs=output_box)

demo.launch()
