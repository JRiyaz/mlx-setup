import argparse
import json
import statistics
import time
import urllib.request

# uv run python scripts/ollama-benchmark.py --models gemma3:4b-it-qat --trials 5 --tokens 500

OLLAMA_URL = "http://127.0.0.1:11434"

PROMPT = """
Explain Python decorators with practical examples.
Cover function decorators, functools.wraps, and a real-world
use case in a FastAPI application. Keep the explanation clear.
"""


def api_request(endpoint, payload=None):
    url = f"{OLLAMA_URL}{endpoint}"

    data = None if payload is None else json.dumps(payload).encode()

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def get_models():
    result = api_request("/api/tags")
    return [model["name"] for model in result.get("models", [])]


def benchmark_model(model, trials, output_tokens):
    print(f"\nBenchmarking: {model}")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "num_predict": output_tokens,
        },
    }

    # Warm-up: loads the model and excludes the first run from results.
    print("Warming up...")
    api_request("/api/chat", payload)

    results = []

    for trial in range(1, trials + 1):
        start = time.perf_counter()
        response = api_request("/api/chat", payload)
        elapsed = time.perf_counter() - start

        prompt_tokens = response.get("prompt_eval_count", 0)
        output_count = response.get("eval_count", 0)

        prompt_duration = response.get("prompt_eval_duration", 0)
        generation_duration = response.get("eval_duration", 0)

        prompt_tps = prompt_tokens / (prompt_duration / 1e9) if prompt_duration else 0

        generation_tps = output_count / (generation_duration / 1e9) if generation_duration else 0

        load_time = response.get("load_duration", 0) / 1e9

        results.append(
            {
                "prompt_tps": prompt_tps,
                "generation_tps": generation_tps,
                "elapsed": elapsed,
                "load_time": load_time,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_count,
            }
        )

        print(
            f"Trial {trial}: "
            f"prompt_tps={prompt_tps:.2f}, "
            f"generation_tps={generation_tps:.2f}, "
            f"total_time={elapsed:.2f}s, "
            f"load_time={load_time:.2f}s"
        )

    def median(key):
        return statistics.median(item[key] for item in results)

    summary = {
        "model": model,
        "prompt_tps": median("prompt_tps"),
        "generation_tps": median("generation_tps"),
        "total_time": median("elapsed"),
        "load_time": median("load_time"),
        "prompt_tokens": median("prompt_tokens"),
        "output_tokens": median("output_tokens"),
    }

    print(
        f"\nMedian: prompt_tps={summary['prompt_tps']:.2f}, "
        f"generation_tps={summary['generation_tps']:.2f}, "
        f"total_time={summary['total_time']:.2f}s"
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    available_models = get_models()
    selected_models = args.models or available_models

    print(f"Found {len(available_models)} Ollama models")

    summaries = []

    for model in selected_models:
        if model not in available_models:
            print(f"\nSkipping unavailable model: {model}")
            continue

        try:
            summaries.append(benchmark_model(model, args.trials, args.tokens))
        except Exception as error:
            print(f"\nFailed: {model} — {error}")

    print("\n" + "=" * 90)
    print("OLLAMA BENCHMARK SUMMARY (median)")
    print("=" * 90)

    print(f"{'Model':<27} {'Prompt TPS':>12} {'Generation TPS':>16} {'Total (s)':>12} {'Load (s)':>10}")

    for item in sorted(
        summaries,
        key=lambda result: result["generation_tps"],
        reverse=True,
    ):
        print(
            f"{item['model']:<27} "
            f"{item['prompt_tps']:>12.2f} "
            f"{item['generation_tps']:>16.2f} "
            f"{item['total_time']:>12.2f} "
            f"{item['load_time']:>10.2f}"
        )


if __name__ == "__main__":
    main()
