import gradio as gr
import subprocess

MODEL_OPTIONS = ["llama2", "mistral", "gemma:2b", "deepseek-coder:1.3b"]

TASK_TEMPLATES = {
    "Freeform Prompt": "",
    "Refine Idea/Thesis": "Refine this concept for clarity, logic, and insight: ",
    "Generate App Skeleton": "Generate the basic code for a web app that does the following: ",
    "Create Visual Art Prompt": "Create a detailed prompt for an AI art generator to produce: ",
    "Simulate Multi-Agent Discussion": "Simulate a conversation between 3 AI agents with distinct goals about: ",
    "Summarize or Index Notes": "Summarize and extract key concepts from the following notes: "
}

def query_ollama_gradio(model, prompt):
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

def set_prompt_from_task(task):
    return TASK_TEMPLATES.get(task, "")

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Local AI Agent Studio")
    gr.Markdown("Choose a task, model, and enter your prompt to run it locally via Ollama.")

    with gr.Row():
        model_selector = gr.Dropdown(choices=MODEL_OPTIONS, value="llama2", label="Select Model")
        task_selector = gr.Dropdown(choices=list(TASK_TEMPLATES.keys()), value="Freeform Prompt", label="Select Task Type")

    prompt_input = gr.Textbox(lines=4, placeholder="Enter your prompt here...", label="Prompt")
    task_selector.change(fn=set_prompt_from_task, inputs=task_selector, outputs=prompt_input)

    run_button = gr.Button("Run Agent")
    output_box = gr.Textbox(label="Response", lines=6)

    run_button.click(fn=query_ollama_gradio, inputs=[model_selector, prompt_input], outputs=output_box)

demo.launch()
