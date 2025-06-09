import subprocess
import argparse
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
        return f"Error: {e}"

def log_response(model, original_input, response, log_file):
    timestamp = datetime.datetime.now().isoformat()
    with open(log_file, "a") as f:
        f.write(f"\n[{timestamp}] [Model: {model}]\n")
        f.write(f"Idea: {original_input}\n\nResponse:\n{response}\n")
        f.write(f"{'-'*50}\n")

def main():
    parser = argparse.ArgumentParser(description="Run Thinking Partner Agent via Ollama.")
    parser.add_argument("--model", type=str, default="llama2", help="Model to use")
    parser.add_argument("--log", type=str, default="thinking_partner_log.txt", help="Log file name")
    args = parser.parse_args()

    print(f"🤖 Thinking Partner is active (Model: {args.model})")
    print("Type 'exit' to quit.\n")

    while True:
        user_idea = input("🧠 Idea: ")
        if user_idea.strip().lower() == "exit":
            break

        prompt = build_prompt(user_idea)
        response = query_ollama(args.model, prompt)
        print(f"\n🗣️ Agent Response:\n{response}\n")
        log_response(args.model, user_idea, response, args.log)

if __name__ == "__main__":
    main()
