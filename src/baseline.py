from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .evaluate import get_paths, load_jsonl

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.pipeline import Pipeline
except ImportError as error:  # pragma: no cover - exercised by environment without sklearn
    SKLEARN_IMPORT_ERROR = error
    Pipeline = None  # type: ignore[assignment]
else:
    SKLEARN_IMPORT_ERROR = None


TARGET_FIELDS = ("issue_type", "product_area")


@dataclass(frozen=True)
class Prediction:
    id: str
    true: str
    predicted: str


def require_sklearn() -> None:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "scikit-learn is required for the baseline. Install it with "
            "`pip install scikit-learn`."
        ) from SKLEARN_IMPORT_ERROR


def ticket_text(ticket: dict[str, Any]) -> str:
    """Build a single text input from a ticket for the text baseline."""
    return f"{ticket['subject']}\n{ticket['body']}"


def build_model(random_state: int = 0):  # type: ignore[no-untyped-def]
    require_sklearn()
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    analyzer="char_wb",
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _labels_for(target: str, labels: Sequence[dict[str, Any]]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for label in labels:
        value = label[target]
        if value not in seen:
            seen.add(value)
            order.append(value)
    return sorted(order)


def train_and_predict(
    train_tickets: Sequence[dict[str, Any]],
    train_labels: Sequence[dict[str, Any]],
    test_tickets: Sequence[dict[str, Any]],
    test_labels: Sequence[dict[str, Any]],
    *,
    target: str,
    random_state: int = 0,
) -> tuple[Any, list[Prediction], float]:
    require_sklearn()
    train_by_id = {ticket["id"]: ticket for ticket in train_tickets}
    train_gold = {label["id"]: label[target] for label in train_labels}
    test_by_id = {ticket["id"]: ticket for ticket in test_tickets}
    test_gold = {label["id"]: label[target] for label in test_labels}

    if set(train_by_id) != set(train_gold):
        raise ValueError("Train tickets and labels must have matching IDs.")
    if set(test_by_id) != set(test_gold):
        raise ValueError("Test tickets and labels must have matching IDs.")

    train_ids = sorted(train_by_id)
    x_train = [ticket_text(train_by_id[ticket_id]) for ticket_id in train_ids]
    y_train = [train_gold[ticket_id] for ticket_id in train_ids]

    model = build_model(random_state=random_state)
    model.fit(x_train, y_train)

    test_ids = sorted(test_by_id)
    x_test = [ticket_text(test_by_id[ticket_id]) for ticket_id in test_ids]
    y_true = [test_gold[ticket_id] for ticket_id in test_ids]

    started_at = time.perf_counter()
    y_pred = model.predict(x_test)
    inference_ms = (time.perf_counter() - started_at) * 1000

    predictions = [
        Prediction(id=ticket_id, true=true, predicted=pred)
        for ticket_id, true, pred in zip(test_ids, y_true, y_pred, strict=True)
    ]
    return model, predictions, inference_ms


def score_predictions(
    predictions: Sequence[Prediction],
    *,
    labels: Sequence[str],
) -> dict[str, Any]:
    require_sklearn()
    y_true = [prediction.true for prediction in predictions]
    y_pred = [prediction.predicted for prediction in predictions]

    correct = sum(1 for prediction in predictions if prediction.true == prediction.predicted)
    total = len(predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {}
    for index, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    matrix = {
        true_label: {
            pred_label: int(cm[i][j])
            for j, pred_label in enumerate(labels)
        }
        for i, true_label in enumerate(labels)
    }
    mismatches = [
        {"id": prediction.id, "predicted": prediction.predicted, "expected": prediction.true}
        for prediction in predictions
        if prediction.predicted != prediction.true
    ]

    return {
        "accuracy": {
            "correct": correct,
            "total": total,
            "accuracy_pct": round(correct / total * 100, 1) if total else 0.0,
        },
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": matrix},
        "mismatches": mismatches,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=False
        ),
    }


def load_llm_accuracy(llm_report_path: Path | None, target: str) -> float | None:
    if llm_report_path is None or not llm_report_path.exists():
        return None
    report = json.loads(llm_report_path.read_text(encoding="utf-8"))
    accuracy = report.get("accuracy", {}).get(target)
    if not accuracy:
        return None
    return float(accuracy["accuracy_pct"])


def run_baseline(
    *,
    train_split: str = "dev",
    test_split: str = "holdout",
    targets: Sequence[str] = TARGET_FIELDS,
    llm_report_path: Path | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    require_sklearn()
    train_tickets_path, train_labels_path = get_paths(train_split)
    test_tickets_path, test_labels_path = get_paths(test_split)
    train_tickets = load_jsonl(train_tickets_path)
    train_labels = load_jsonl(train_labels_path)
    test_tickets = load_jsonl(test_tickets_path)
    test_labels = load_jsonl(test_labels_path)

    if llm_report_path is None and test_split in ("dev", "holdout"):
        candidate = Path("reports") / f"{test_split}_report.json"
        llm_report_path = candidate if candidate.exists() else None

    results: dict[str, Any] = {}
    for target in targets:
        labels = _labels_for(target, train_labels)
        labels.extend(
            label for label in _labels_for(target, test_labels) if label not in labels
        )
        _, predictions, inference_ms = train_and_predict(
            train_tickets,
            train_labels,
            test_tickets,
            test_labels,
            target=target,
            random_state=random_state,
        )
        scored = score_predictions(predictions, labels=labels)
        llm_accuracy = load_llm_accuracy(llm_report_path, target)
        scored["inference_ms"] = {
            "total": round(inference_ms, 3),
            "per_ticket": round(inference_ms / len(predictions), 3) if predictions else 0.0,
        }
        scored["llm_accuracy_pct"] = llm_accuracy
        scored["delta_pct_points"] = (
            round(scored["accuracy"]["accuracy_pct"] - llm_accuracy, 1)
            if llm_accuracy is not None
            else None
        )
        results[target] = scored

    train_counts = {
        target: Counter(label[target] for label in train_labels) for target in targets
    }
    test_counts = {
        target: Counter(label[target] for label in test_labels) for target in targets
    }

    return {
        "train_split": train_split,
        "test_split": test_split,
        "train_size": len(train_tickets),
        "test_size": len(test_tickets),
        "model": "TfidfVectorizer(char_wb 1-2) + LogisticRegression(class_weight=balanced)",
        "random_state": random_state,
        "targets": results,
        "train_label_counts": {
            target: dict(sorted(counts.items())) for target, counts in train_counts.items()
        },
        "test_label_counts": {
            target: dict(sorted(counts.items())) for target, counts in test_counts.items()
        },
        "llm_report_used": str(llm_report_path) if llm_report_path else None,
        "notes": [
            "Baseline is evaluated as a side-by-side classifier, not wired into triage.",
            "Training uses only the train split; the test split is untouched.",
            "With the current 40/16 dataset this is an illustrative small-data baseline.",
        ],
    }


def print_report(report: dict[str, Any]) -> None:
    print(
        f"Baseline: {report['model']} (train={report['train_split']}, "
        f"test={report['test_split']}, n={report['train_size']}->{report['test_size']})"
    )
    if report.get("llm_report_used"):
        print(f"LLM comparison: {report['llm_report_used']}")
    for target, scores in report["targets"].items():
        accuracy = scores["accuracy"]
        print(f"\n{target}:")
        print(
            f"  accuracy: {accuracy['correct']}/{accuracy['total']} "
            f"({accuracy['accuracy_pct']:.1f}%)"
        )
        print(f"  macro_f1: {scores['macro_f1']:.4f}")
        print(f"  weighted_f1: {scores['weighted_f1']:.4f}")
        print(
            f"  inference_ms: total {scores['inference_ms']['total']}, "
            f"per_ticket {scores['inference_ms']['per_ticket']}"
        )
        if scores["llm_accuracy_pct"] is not None:
            delta = scores["delta_pct_points"]
            sign = "+" if delta >= 0 else ""
            print(
                f"  llm_accuracy: {scores['llm_accuracy_pct']:.1f}% "
                f"(delta {sign}{delta:.1f} pp)"
            )
        for mismatch in scores["mismatches"]:
            print(
                f"    {mismatch['id']}: {mismatch['predicted']!r} != {mismatch['expected']!r}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    from .evaluate import SPLITS

    parser = argparse.ArgumentParser(description="Evaluate a scikit-learn baseline.")
    split_choices = tuple(sorted(SPLITS))
    default_train = "train" if "train" in SPLITS else "dev"
    default_test = "test" if "test" in SPLITS else "holdout"
    parser.add_argument("--train-split", default=default_train, choices=split_choices)
    parser.add_argument("--test-split", default=default_test, choices=split_choices)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TARGET_FIELDS),
        choices=list(TARGET_FIELDS),
    )
    parser.add_argument("--llm-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    if args.train_split == args.test_split:
        parser.error("--train-split and --test-split must differ.")

    report = run_baseline(
        train_split=args.train_split,
        test_split=args.test_split,
        targets=args.targets,
        llm_report_path=args.llm_report,
        random_state=args.random_state,
    )
    print_report(report)

    output = args.output or Path("reports") / f"baseline_{args.test_split}_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport written to {output}")


if __name__ == "__main__":
    main()
