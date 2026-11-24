from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .llm_client import DEFAULT_MODEL, LLMExtractor
from .triage import TriagePipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TICKETS_PATH = PROJECT_ROOT / "data" / "tickets.jsonl"
LABELS_PATH = PROJECT_ROOT / "data" / "gold_labels.jsonl"
HOLDOUT_TICKETS_PATH = PROJECT_ROOT / "data" / "holdout_tickets.jsonl"
HOLDOUT_LABELS_PATH = PROJECT_ROOT / "data" / "holdout_labels.jsonl"
TRAIN_TICKETS_PATH = PROJECT_ROOT / "data" / "train_tickets.jsonl"
TRAIN_LABELS_PATH = PROJECT_ROOT / "data" / "train_labels.jsonl"
TEST_TICKETS_PATH = PROJECT_ROOT / "data" / "test_tickets.jsonl"
TEST_LABELS_PATH = PROJECT_ROOT / "data" / "test_labels.jsonl"
SPLITS = {
    "dev": (TICKETS_PATH, LABELS_PATH),
    "holdout": (HOLDOUT_TICKETS_PATH, HOLDOUT_LABELS_PATH),
    "train": (TRAIN_TICKETS_PATH, TRAIN_LABELS_PATH),
    "test": (TEST_TICKETS_PATH, TEST_LABELS_PATH),
}

INPUT_USD_PER_1M_TOKENS = 0.15
OUTPUT_USD_PER_1M_TOKENS = 0.60

FIELDS_TO_SCORE = {
    "language": "language",
    "issue_type": "issue_type",
    "priority": "priority",
    "product_area": "product_area",
    "escalation_required": "escalation_required",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_paths(split: str) -> tuple[Path, Path]:
    try:
        return SPLITS[split]
    except KeyError as error:
        raise ValueError(f"Unknown split {split!r}; choose from {sorted(SPLITS)}") from error


def _percentile_50(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return statistics.median(ordered)


def run(
    pipeline: TriagePipeline,
    tickets: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
    *,
    split: str = "dev",
    input_usd_per_1m_tokens: float = INPUT_USD_PER_1M_TOKENS,
    output_usd_per_1m_tokens: float = OUTPUT_USD_PER_1M_TOKENS,
) -> dict[str, Any]:
    """Score a pipeline against gold labels and return a JSON-serializable report."""
    labels_by_id = {label["id"]: label for label in labels}
    ticket_ids = [ticket["id"] for ticket in tickets]

    if len(ticket_ids) != len(set(ticket_ids)):
        raise ValueError("Ticket IDs must be unique.")
    if not set(ticket_ids).issubset(labels_by_id):
        raise ValueError("Some ticket IDs have no gold label.")

    correct: dict[str, int] = {field: 0 for field in FIELDS_TO_SCORE}
    mismatches: dict[str, list[dict[str, Any]]] = {
        field: [] for field in FIELDS_TO_SCORE
    }
    latencies_ms: list[float] = []
    input_tokens = 0
    output_tokens = 0

    for index, ticket in enumerate(tickets, start=1):
        started_at = time.perf_counter()
        result = pipeline.triage(ticket)
        latencies_ms.append((time.perf_counter() - started_at) * 1000)

        label = labels_by_id[ticket["id"]]
        prediction = result.as_dict()

        for prediction_field, label_field in FIELDS_TO_SCORE.items():
            if prediction[prediction_field] == label[label_field]:
                correct[prediction_field] += 1
            else:
                mismatches[prediction_field].append(
                    {
                        "id": ticket["id"],
                        "predicted": prediction[prediction_field],
                        "expected": label[label_field],
                    }
                )

        input_tokens += result.prompt_tokens
        output_tokens += result.completion_tokens
        print(f"Processed {index}/{len(tickets)}: {ticket['id']}")

    total = len(tickets)
    accuracy = {
        field: {
            "correct": correct[field],
            "total": total,
            "accuracy_pct": round(correct[field] / total * 100, 1),
            "mismatches": mismatches[field],
        }
        for field in FIELDS_TO_SCORE
    }

    estimated_cost_usd = (
        input_tokens * input_usd_per_1m_tokens
        + output_tokens * output_usd_per_1m_tokens
    ) / 1_000_000

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "model": getattr(pipeline.extractor, "model", "injected extractor"),
        "evaluated": total,
        "accuracy": accuracy,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 1),
            "p50": round(_percentile_50(latencies_ms), 1),
            "max": round(max(latencies_ms), 1),
        },
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "input_usd_per_1m": input_usd_per_1m_tokens,
            "output_usd_per_1m": output_usd_per_1m_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
        },
        "summary_scored": False,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\nSplit: {report['split']}")
    print(f"Model: {report['model']}")
    print(f"Tickets evaluated: {report['evaluated']}")
    print("\nAccuracy:")
    for field, scores in report["accuracy"].items():
        print(
            f"  {field}: {scores['correct']}/{scores['total']} "
            f"({scores['accuracy_pct']:.1f}%)"
        )
        for mismatch in scores["mismatches"]:
            print(
                f"    {mismatch['id']}: {mismatch['predicted']!r} "
                f"!= {mismatch['expected']!r}"
            )

    latency = report["latency_ms"]
    tokens = report["tokens"]
    print(
        f"\nLatency (ms): mean {latency['mean']}, p50 {latency['p50']}, "
        f"max {latency['max']}"
    )
    print(f"Input tokens: {tokens['input']}")
    print(f"Output tokens: {tokens['output']}")
    print(f"Estimated API cost: ${tokens['estimated_cost_usd']:.6f}")
    print("Summary accuracy: not scored (gold labels contain no summaries).")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        default="dev",
        help="dev is the 40-ticket tuning set; holdout is 16 unseen tickets.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to reports/<split>_report.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N tickets (useful for a quick smoke test).",
    )
    args = parser.parse_args(argv)

    tickets_path, labels_path = get_paths(args.split)
    tickets = load_jsonl(tickets_path)
    labels = load_jsonl(labels_path)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        tickets = tickets[: args.limit]

    pipeline = TriagePipeline(LLMExtractor(model=args.model))
    report = run(pipeline, tickets, labels, split=args.split)
    print_report(report)

    output = args.output or PROJECT_ROOT / "reports" / f"{args.split}_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nReport written to {output}")


if __name__ == "__main__":
    main()
