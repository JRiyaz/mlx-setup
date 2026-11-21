
import subprocess
import statistics
import time

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1976/v1",
    api_key="local",
)

PROMPT = """
Explain Python decorators with practical examples.
Cover function decorators, functools.wraps, and a
real-world FastAPI use case. Aim for approximately
500 tokens in your response.
"""

TRIALS = 5


def count_tokens(text):
    result = subprocess.run(
        ["fm", "count-tokens", "--quiet", text],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def generate():
    start = time.perf_counter()

    response = client.chat.completions.create(
        model="system",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0,
        stream=False,
    )

    elapsed = time.perf_counter() - start

    content = response.choices[0].message.content or ""

    # Prefer token usage reported by Apple's API.
    usage = response.usage
    output_tokens = (
        usage.completion_tokens
        if usage and usage.completion_tokens
        else count_tokens(content)
    )

    return elapsed, output_tokens, content


def main():
    prompt_tokens = count_tokens(PROMPT)

    print("Warming up...")
    generate()

    results = []

    for trial in range(1, TRIALS + 1):
        elapsed, tokens, _ = generate()

        tps = tokens / elapsed if elapsed else 0

        results.append({
            "time": elapsed,
            "tokens": tokens,
            "tps": tps,
        })

        print(
            f"Trial {trial}: "
            f"tokens={tokens}, "
            f"time={elapsed:.2f}s, "
            f"generation_tps={tps:.2f}"
        )

    print("\n========== SUMMARY ==========")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Median response time: "
          f"{statistics.median(r['time'] for r in results):.2f}s")
    print(f"Median output tokens: "
          f"{statistics.median(r['tokens'] for r in results):.0f}")
    print(f"Median generation TPS: "
          f"{statistics.median(r['tps'] for r in results):.2f}")


if __name__ == "__main__":
    main()

