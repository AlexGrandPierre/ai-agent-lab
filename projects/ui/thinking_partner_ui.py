import gradio as gr
import subprocess
import datetime

def build_prompt(user_idea):
    return f"""
You are a philosophical thinking partner. Engage with the following idea by doing three things:
1. Clarify the core claim in your own words.
2. Offer one reasonable counterpoint or alternative interpretation.
3. Suggest a more precise or rigorous formulation of the idea.

Here is the idea:
\"\"\"{user_idea}\"\"\"
"""

def query_ollama(model, prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Error querying Ollama: {e}"

def log_interaction(model, idea, response):
    timestamp = datetime.datetime.now().isoformat()
    with open("thinking_partner_ui_log.txt", "a") as f:
        f.write(f"\n[{timestamp}] [Model: {model}]\nIdea: {idea}\n\nResponse:\n{response}\n{'-'*50}\n")

def run_thinking_partner(idea, model):
    prompt = build_prompt(idea)
    response = query_ollama(model, prompt)
    log_interaction(model, idea, response)
    return response

with gr.Blocks(title="Thinking Partner Agent") as demo:
    gr.Markdown("## 🤖 Thinking Partner\nRefine philosophical or conceptual ideas with an AI collaborator.")
    
    with gr.Row():
        model_choice = gr.Dropdown(["llama2", "mistral", "gemma:2b"], value="llama2", label="Choose Model")
    
    idea_input = gr.Textbox(label="Your Idea", placeholder="Enter a concept, thesis, or claim...")
    
    output = gr.Textbox(label="Agent Response", lines=12)

    submit_btn = gr.Button("Refine My Idea")

    submit_btn.click(fn=run_thinking_partner, inputs=[idea_input, model_choice], outputs=output)

demo.launch()
