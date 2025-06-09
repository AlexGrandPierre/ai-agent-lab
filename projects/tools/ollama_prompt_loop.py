import subprocess
import datetime

MODEL_NAME = "llama2"  # Change to "mistral", "gemma:2b", etc. as needed
LOG_FILE = "ollama_chat_log.txt"

def query_ollama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8")
    except Exception as e:
        return f"Error querying Ollama: {e}"

def log_interaction(prompt, response):
    timestamp = datetime.datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{timestamp}]\nUser: {prompt}\nAgent: {response}\n{'-'*40}")

def main():
    print(f"Running Ollama agent with model: {MODEL_NAME}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        response = query_ollama(user_input)
        print(f"\nAgent: {response}\n")
        log_interaction(user_input, response)

if __name__ == "__main__":
    main()
