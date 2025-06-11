import gradio as gr
import subprocess

MODEL_OPTIONS = ["llama2", "mistral", "gemma:2b", "deepseek-coder:1.3b"]

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

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Local AI Agent Studio")
    gr.Markdown("Enter your prompt, choose a model, and run it locally with Ollama.")

    with gr.Row():
        model_selector = gr.Dropdown(choices=MODEL_OPTIONS, value="llama2", label="Select Model")
        prompt_input = gr.Textbox(lines=4, placeholder="Enter your prompt here...", label="Prompt")

    run_button = gr.Button("Run Agent")
    output_box = gr.Textbox(label="Response")

    run_button.click(fn=query_ollama_gradio, inputs=[model_selector, prompt_input], outputs=output_box)

demo.launch()
