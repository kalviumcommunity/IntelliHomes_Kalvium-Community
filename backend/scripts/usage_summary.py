"""Generate a usage summary from the RAG JSONL request log.

Outputs a JSON summary with totals per day and overall aggregates.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "rag_requests.jsonl"
OUT_PATH = Path(__file__).resolve().parents[1] / "logs" / "usage_summary.json"


def parse_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return "unknown"


def build_summary():
    if not LOG_PATH.exists():
        print("No logs found at", LOG_PATH)
        return {}

    per_day = defaultdict(lambda: {"requests": 0, "cache_hits": 0, "cost_usd": 0.0, "tokens": 0, "latency_ms": 0, "errors": 0})
    overall = {"requests": 0, "cache_hits": 0, "cost_usd": 0.0, "tokens": 0, "latency_ms": 0, "errors": 0}

    with open(LOG_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            day = parse_timestamp(rec.get("timestamp", ""))
            per_day[day]["requests"] += 1
            overall["requests"] += 1
            if rec.get("cache_hit"):
                per_day[day]["cache_hits"] += 1
                overall["cache_hits"] += 1
            if rec.get("cost"):
                per_day[day]["cost_usd"] += float(rec.get("cost", 0.0))
                overall["cost_usd"] += float(rec.get("cost", 0.0))
            if rec.get("tokens"):
                per_day[day]["tokens"] += int(rec.get("tokens", 0))
                overall["tokens"] += int(rec.get("tokens", 0))
            if rec.get("latency_ms"):
                per_day[day]["latency_ms"] += int(rec.get("latency_ms", 0))
                overall["latency_ms"] += int(rec.get("latency_ms", 0))
            if rec.get("error"):
                per_day[day]["errors"] += 1
                overall["errors"] += 1

    # Post-process averages
    for day, stats in per_day.items():
        if stats["requests"]:
            stats["avg_latency_ms"] = stats["latency_ms"] / stats["requests"]
        else:
            stats["avg_latency_ms"] = 0

    if overall["requests"]:
        overall["avg_latency_ms"] = overall["latency_ms"] / overall["requests"]
    else:
        overall["avg_latency_ms"] = 0

    summary = {"per_day": dict(per_day), "overall": overall}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Wrote summary to", OUT_PATH)
    return summary


if __name__ == "__main__":
    build_summary()
