from litellm import completion

resp = completion(
    model="together_ai/moonshotai/Kimi-K2.5",
    messages=[
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {"role": "user", "content": "Say hello."}
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=32,
    reasoning={"enabled": False},
)
print(resp)