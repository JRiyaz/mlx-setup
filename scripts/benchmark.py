import time

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

prompt = "Explain Python decorators with practical examples."

start = time.perf_counter()

response = client.chat.completions.create(
    model="mlx-local",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=500,
)

elapsed = time.perf_counter() - start

answer = response.choices[0].message.content
usage = response.usage

print("\nResponse:")
print(answer)

print("\nPerformance:")
print(f"Total time: {elapsed:.2f} seconds")

if usage:
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Generated tokens: {usage.completion_tokens}")

    tps = usage.completion_tokens / elapsed
    print(f"End-to-end tokens/sec: {tps:.2f}")
else:
    print("Token usage was not returned by the server.")
