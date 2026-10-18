
import ollama

MODEL = "gemma4:e4b"

response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a legal document analysis assistant. "
                "Explain legal concepts in simple language. "
                "Do not invent facts or claim to replace a lawyer."
            )
        },
        {
            "role": "user",
            "content": (
                "Explain a contract termination clause. "
                "Give one example and identify one potential risk."
            )
        }
    ]
)

print(response["message"]["content"])

