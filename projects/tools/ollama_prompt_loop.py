import subprocess
import datetime
import argparse

def query_ollama(model, prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8")
    except Exception as e:
        return f"Error querying Ollama: {e}"

def log_interaction(model, prompt, response, log_file):
    timestamp = datetime.datetime.now().isoformat()
    with open(log_file, "a") as f:
        f.write(f"\n[{timestamp}] [Model: {model}]\nUser: {prompt}\nAgent: {response}\n{'-'*40}\n")

def main():
    parser = argparse.ArgumentParser(description="Run local LLM agent via Ollama.")
    parser.add_argument("--model", type=str, default="llama2", help="Model name (e.g. llama2, mistral, gemma:2b)")
    parser.add_argument("--log", type=str, default="ollama_chat_log.txt", help="Log file to save chat history")
    args = parser.parse_args()

    print(f"\nRunning Ollama agent with model: {args.model}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        response = query_ollama(args.model, user_input)
        print(f"\nAgent: {response.strip()}\n")
        log_interaction(args.model, user_input, response, args.log)

if __name__ == "__main__":
    main()
