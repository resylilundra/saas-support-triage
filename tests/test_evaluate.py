import json

import pytest

from src import evaluate
from src.llm_client import Extraction
from src.triage import TriagePipeline


def build_pipeline(tickets, labels, overrides=None):
    labels_by_id = {label["id"]: label for label in labels}
    fields_by_subject = {
        ticket["subject"]: labels_by_id[ticket["id"]] for ticket in tickets
    }
    overrides = overrides or {}

    class LabelExtractor:
        model = "fake-model"

        def extract(self, subject, body, *, feedback=None):
            label = fields_by_subject[subject]
            fields = {
                "language": label["language"],
                "issue_type": label["issue_type"],
                "priority": label["priority"],
                "product_area": label["product_area"],
                "summary": "Generated summary.",
            }
            fields.update(overrides.get(label["id"], {}))
            return Extraction(json.dumps(fields), 10, 5)

    return TriagePipeline(LabelExtractor())


def test_run_scores_all_fields_and_cost():
    tickets = evaluate.load_jsonl(evaluate.TICKETS_PATH)
    labels = evaluate.load_jsonl(evaluate.LABELS_PATH)
    pipeline = build_pipeline(tickets, labels)

    report = evaluate.run(pipeline, tickets, labels)

    assert report["evaluated"] == 40
    assert report["model"] == "fake-model"
    for scores in report["accuracy"].values():
        assert scores["correct"] == 40
        assert scores["mismatches"] == []
    expected_cost = (
        400 * evaluate.INPUT_USD_PER_1M_TOKENS
        + 200 * evaluate.OUTPUT_USD_PER_1M_TOKENS
    ) / 1_000_000
    assert report["tokens"] == {
        "input": 400,
        "output": 200,
        "input_usd_per_1m": evaluate.INPUT_USD_PER_1M_TOKENS,
        "output_usd_per_1m": evaluate.OUTPUT_USD_PER_1M_TOKENS,
        "estimated_cost_usd": round(expected_cost, 6),
    }
    assert set(report["latency_ms"]) == {"mean", "p50", "max"}
    assert report["summary_scored"] is False


def test_run_records_mismatches():
    tickets = evaluate.load_jsonl(evaluate.TICKETS_PATH)
    labels = evaluate.load_jsonl(evaluate.LABELS_PATH)
    pipeline = build_pipeline(tickets, labels, {"t01": {"priority": "low"}})

    report = evaluate.run(pipeline, tickets, labels)

    priority_scores = report["accuracy"]["priority"]
    assert priority_scores["correct"] == 39
    assert priority_scores["accuracy_pct"] == 97.5
    assert priority_scores["mismatches"] == [
        {"id": "t01", "predicted": "low", "expected": "normal"}
    ]


def test_run_rejects_tickets_without_labels():
    tickets = evaluate.load_jsonl(evaluate.TICKETS_PATH)[:2]
    labels = evaluate.load_jsonl(evaluate.LABELS_PATH)[:1]

    try:
        evaluate.run(build_pipeline([], []), tickets, labels)
    except ValueError as error:
        assert "no gold label" in str(error)
    else:
        raise AssertionError("Expected a ValueError for a missing gold label")


def test_get_paths_selects_splits():
    assert evaluate.get_paths("dev") == (evaluate.TICKETS_PATH, evaluate.LABELS_PATH)
    assert evaluate.get_paths("holdout") == (
        evaluate.HOLDOUT_TICKETS_PATH,
        evaluate.HOLDOUT_LABELS_PATH,
    )
    with pytest.raises(ValueError, match="Unknown split"):
        evaluate.get_paths("unknown")


def test_main_writes_report(monkeypatch, tmp_path):
    tickets = evaluate.load_jsonl(evaluate.TICKETS_PATH)
    labels = evaluate.load_jsonl(evaluate.LABELS_PATH)
    monkeypatch.setattr(
        evaluate,
        "TriagePipeline",
        lambda extractor: build_pipeline(tickets, labels),
    )
    output = tmp_path / "nested" / "report.json"

    evaluate.main(["--limit", "2", "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluated"] == 2
    assert report["accuracy"]["language"]["correct"] == 2
    assert report["split"] == "dev"


def test_main_writes_holdout_report(monkeypatch, tmp_path):
    tickets = evaluate.load_jsonl(evaluate.HOLDOUT_TICKETS_PATH)
    labels = evaluate.load_jsonl(evaluate.HOLDOUT_LABELS_PATH)
    monkeypatch.setattr(
        evaluate,
        "TriagePipeline",
        lambda extractor: build_pipeline(tickets, labels),
    )
    output = tmp_path / "holdout.json"

    evaluate.main(["--split", "holdout", "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["split"] == "holdout"
    assert report["evaluated"] == 16
    for scores in report["accuracy"].values():
        assert scores["correct"] == 16
