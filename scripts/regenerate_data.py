#!/usr/bin/env python3
"""Regenerar dashboard.json e CSV com a nova lógica de timeout."""

import csv
import json
import sys
from pathlib import Path

REPORTS = Path(__file__).parent.parent / "reports"
CSV_PATH = REPORTS / "benchmark_results.csv"
DASH_PATH = REPORTS / "benchmark_results.dashboard.json"


def build_rows_from_csv(csv_path: Path) -> list[dict]:
    """Ler CSV e retornar lista de dicts (simulando BenchmarkResult)."""
    rows: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "model": row["model"],
                "model_path": row["model_path"],
                "model_size_bytes": int(row["model_size_bytes"]),
                "model_size_gib": float(row["model_size_gib"]),
                "context_size": int(row["context_size"]),
                "server_ctx_train": int(row["server_ctx_train"]),
                "observed_state_size_bytes": int(row["observed_state_size_bytes"]),
                "observed_state_size_gib": float(row["observed_state_size_gib"]),
                "estimated_kv_cache_bytes": int(row["estimated_kv_cache_bytes"]),
                "estimated_kv_cache_gib": float(row["estimated_kv_cache_gib"]),
                "estimated_loaded_bytes": int(row["estimated_loaded_bytes"]),
                "estimated_loaded_gib": float(row["estimated_loaded_gib"]),
                "prompt_id": row["prompt_id"],
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "tokens_generated": int(row["tokens_generated"]),
                "tokens_per_second": float(row["tokens_per_second"]),
                "return_code": int(row["return_code"]),
                "output": row["output"],
                "error": row["error"],
                "timeout": row.get("timeout", "false").lower() == "true",
                "category": "",
            })
    return rows


def write_dashboard_json(path: Path, rows: list[dict]) -> None:
    """Gerar dashboard JSON com timeout (sem categoria)."""
    categories = {"coding", "extraction", "instruction", "reasoning"}
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    models = []
    for model, model_rows in grouped.items():
        sample = model_rows[0]
        prompts_tested = len(model_rows)
        successful = sum(1 for row in model_rows if row["return_code"] == 0)
        avg_tps = sum(row["tokens_per_second"] for row in model_rows) / prompts_tested if prompts_tested else 0.0
        weighted_tps = (
            sum(row["tokens_generated"] * row["elapsed_seconds"] for row in model_rows)
            / sum(row["elapsed_seconds"] for row in model_rows)
            if any(row["elapsed_seconds"] > 0 for row in model_rows)
            else 0.0
        )
        timeout_count = sum(1 for row in model_rows if row["timeout"])

        entry = {
            "model": model,
            "model_path": sample["model_path"],
            "model_size_bytes": sample["model_size_bytes"],
            "model_size_gib": round(sample["model_size_gib"], 4),
            "context_size": sample["context_size"],
            "server_ctx_train": sample["server_ctx_train"],
            "observed_state_size_bytes": sample["observed_state_size_bytes"],
            "observed_state_size_gib": round(sample["observed_state_size_gib"], 4),
            "estimated_kv_cache_bytes": sample["estimated_kv_cache_bytes"],
            "estimated_kv_cache_gib": round(sample["estimated_kv_cache_gib"], 4),
            "estimated_loaded_bytes": sample["estimated_loaded_bytes"],
            "estimated_loaded_gib": round(sample["estimated_loaded_gib"], 4),
            "avg_tokens_per_second": round(avg_tps, 4),
            "prompts_tested": prompts_tested,
            "successful_prompts": successful,
            "categories": {
                cat: {
                    "prompts_tested": len([
                        r for r in model_rows if r["category"] == cat
                    ]),
                    "successful_prompts": sum(
                        1
                        for r in model_rows
                        if r["category"] == cat and r["return_code"] == 0
                    ),
                    "timeout_count": sum(1 for r in model_rows if r["category"] == cat and r["timeout"]),
                    "timeout_badge": "⏱️" if sum(1 for r in model_rows if r["category"] == cat and r["timeout"]) > 0 else "",
                }
                for cat in categories
            },
            "timeout_count": timeout_count,
            "timeout_badge": "⏱️" if timeout_count > 0 else "",
        }

        entry["avg_tokens_per_second_weighted"] = round(weighted_tps, 4)
        entry["timeout_badge"] = "⏱️" if timeout_count > 0 else ""
        entry["min_tokens_per_second"] = min(
            (r["tokens_per_second"] for r in model_rows), default=0.0
        )
        entry["max_tokens_per_second"] = max(
            (r["tokens_per_second"] for r in model_rows), default=0.0
        )
        entry["avg_elapsed_seconds"] = round(
            sum(r["elapsed_seconds"] for r in model_rows) / len(model_rows), 4
        ) if model_rows else 0.0
        entry["avg_tokens_generated"] = round(
            sum(r["tokens_generated"] for r in model_rows) / len(model_rows), 2
        ) if model_rows else 0.0

        models.append(entry)

    payload = {
        "models": sorted(models, key=lambda i: i["avg_tokens_per_second"], reverse=True),
        "category_summaries": {},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Dashboard salvo em: {path.resolve()}")


def write_csv_with_timeout(path: Path, rows: list[dict]) -> None:
    """Reescrever CSV incluindo coluna timeout."""
    header = [
        "model",
        "model_path",
        "model_size_bytes",
        "model_size_gib",
        "context_size",
        "server_ctx_train",
        "observed_state_size_bytes",
        "observed_state_size_gib",
        "estimated_kv_cache_bytes",
        "estimated_kv_cache_gib",
        "estimated_loaded_bytes",
        "estimated_loaded_gib",
        "prompt_id",
        "elapsed_seconds",
        "tokens_generated",
        "tokens_per_second",
        "return_code",
        "output",
        "error",
        "timeout",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["model_path"],
                    row["model_size_bytes"],
                    row["model_size_gib"],
                    row["context_size"],
                    row["server_ctx_train"],
                    row["observed_state_size_bytes"],
                    row["observed_state_size_gib"],
                    row["estimated_kv_cache_bytes"],
                    row["estimated_kv_cache_gib"],
                    row["estimated_loaded_bytes"],
                    row["estimated_loaded_gib"],
                    row["prompt_id"],
                    f"{row['elapsed_seconds']:.4f}",
                    row["tokens_generated"],
                    f"{row['tokens_per_second']:.4f}",
                    row["return_code"],
                    row["output"],
                    row["error"],
                    "true" if row["timeout"] else "false",
                ]
            )
    print(f"CSV salvo em: {path.resolve()}")


if __name__ == "__main__":
    rows = build_rows_from_csv(CSV_PATH)
    print(f"Lidos {len(rows)} linhas do CSV")
    write_csv_with_timeout(CSV_PATH, rows)
    write_dashboard_json(DASH_PATH, rows)
