"""Benchmark inference latency (/predict) with warm-up and server/client mean, p50, p95."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def build_multipart_body(file_path: Path, field_name: str = "file") -> tuple[bytes, str]:
    boundary = f"----KWSBoundary{uuid.uuid4().hex}"
    file_bytes = file_path.read_bytes()
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + file_bytes + footer, f"multipart/form-data; boundary={boundary}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--file", default="samples/yes.wav", type=Path)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not args.file.is_file():
        raise SystemExit(f"WAV file not found: {args.file}")

    body, content_type = build_multipart_body(args.file)
    endpoint = f"{args.url.rstrip('/')}/predict"

    server_latencies: list[float] = []
    client_latencies: list[float] = []
    total_requests = args.warmup + args.requests

    for index in range(total_requests):
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if index >= args.warmup:
            client_latencies.append(elapsed_ms)
            server_latencies.append(float(payload["latency_ms"]))

    result = {
        "file": str(args.file),
        "warmup": args.warmup,
        "requests": args.requests,
        "server_ms": {
            "mean": round(statistics.mean(server_latencies), 2),
            "p50": round(percentile(server_latencies, 0.50), 2),
            "p95": round(percentile(server_latencies, 0.95), 2),
        },
        "client_ms": {
            "mean": round(statistics.mean(client_latencies), 2),
            "p50": round(percentile(client_latencies, 0.50), 2),
            "p95": round(percentile(client_latencies, 0.95), 2),
        },
    }
    output = json.dumps(result, indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
