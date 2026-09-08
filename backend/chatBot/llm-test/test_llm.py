from openai import OpenAI

print("Starting LLM test...")

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

print("Sending request to LM Studio...")

response = client.chat.completions.create(
    model="llama-3.2-3b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain what embeddings are in simple terms."
        }
    ],
    temperature=0.7
)

print("\nLLM RESPONSE:\n")
print(response.choices[0].message.content)