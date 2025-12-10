import openai
import mlflow

# Enable auto-tracing for OpenAI (works with DeepSeek)
mlflow.openai.autolog()

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DeepSeek")

# Initialize the OpenAI client with DeepSeek API endpoint
client = openai.OpenAI(
    base_url="https://api.deepseek.com", api_key="sk-246b655a914f4363b3b447075f8e8259"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    temperature=0.1,
    max_tokens=100,
)