"""
rescore.py - Re-pontua um benchmark_results.csv já existente usando o
benchmark_scorer.py atual, sem precisar rodar os modelos de novo.

Uso:
    python rescore.py ..\reports\benchmark_results.csv
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from any directory, not just scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_scorer import BenchmarkEvaluator, print_results

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"

# CSVs gerados antes do fix do clean_response podem ter uma tag de
# linguagem solta na primeira linha (ex: "json\n{...}") porque a cerca
# ```json``` só tinha o "```" removido, não a palavra da linguagem.
_LANG_TAGS = {
    "json", "python", "py", "js", "javascript", "bash", "sh",
    "text", "txt", "yaml", "yml", "xml", "html", "sql", "csv",
}


def strip_leaked_lang_tag(output: str) -> str:
    parts = output.split("\n", 1)
    if len(parts) == 2 and parts[0].strip().lower() in _LANG_TAGS:
        return parts[1]
    return output


def main(csv_path: Path) -> int:
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    prompts_map = {p["id"]: p for p in prompts}

    rows_by_model: dict[str, dict[str, str]] = defaultdict(dict)
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_model[row["model"]][row["prompt_id"]] = strip_leaked_lang_tag(row["output"])

    evaluator = BenchmarkEvaluator()
    all_results = []
    for model, responses in rows_by_model.items():
        evaluated = evaluator.evaluate_model(model, responses, prompts_map)
        print_results(evaluated)
        all_results.append((model, evaluated))

    print("\n" + "=" * 60)
    print("  RESUMO (score final, ordenado)")
    print("=" * 60)
    for model, evaluated in sorted(all_results, key=lambda x: x[1].final_score, reverse=True):
        print(f"  {evaluated.final_score:5.2f}  {model}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python rescore.py <caminho_para_benchmark_results.csv>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1])))
