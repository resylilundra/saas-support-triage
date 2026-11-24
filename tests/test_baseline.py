import json

import pytest

from src import baseline
from src.evaluate import load_jsonl


def _require_sklearn():
    if baseline.SKLEARN_IMPORT_ERROR is not None:
        pytest.skip("scikit-learn is not installed in the test environment.")


def test_ticket_text_includes_subject_and_body():
    ticket = {"id": "x", "subject": "Subject", "body": "Body"}
    assert baseline.ticket_text(ticket) == "Subject\nBody"


def test_train_and_predict_uses_only_provided_splits(monkeypatch):
    _require_sklearn()
    dev_tickets = load_jsonl(baseline.get_paths("dev")[0])[:8]
    dev_labels = load_jsonl(baseline.get_paths("dev")[1])[:8]
    holdout_tickets = load_jsonl(baseline.get_paths("holdout")[0])[:4]
    holdout_labels = load_jsonl(baseline.get_paths("holdout")[1])[:4]

    seen = {"train": None, "test": None}

    class DummyModel:
        def fit(self, x, y):
            seen["train"] = (list(x), list(y))
            return self

        def predict(self, x):
            seen["test"] = list(x)
            return list(seen["train"][1][: len(x)])

    monkeypatch.setattr(baseline, "build_model", lambda random_state=0: DummyModel())

    _, predictions, inference_ms = baseline.train_and_predict(
        dev_tickets,
        dev_labels,
        holdout_tickets,
        holdout_labels,
        target="issue_type",
    )

    assert seen["train"][0] == [baseline.ticket_text(t) for t in dev_tickets]
    sorted_holdout = sorted(holdout_tickets, key=lambda t: t["id"])
    assert seen["test"] == [baseline.ticket_text(t) for t in sorted_holdout]
    assert [p.id for p in predictions] == [t["id"] for t in sorted_holdout]
    assert inference_ms >= 0


def test_run_baseline_reports_metrics_and_comparison(monkeypatch, tmp_path):
    _require_sklearn()
    monkeypatch.chdir(tmp_path)

    dev_tickets = load_jsonl(baseline.get_paths("dev")[0])
    dev_labels = load_jsonl(baseline.get_paths("dev")[1])
    holdout_tickets = load_jsonl(baseline.get_paths("holdout")[0])
    holdout_labels = load_jsonl(baseline.get_paths("holdout")[1])
    gold = {label["id"]: label for label in holdout_labels}

    def fake_train_and_predict(train_t, train_l, test_t, test_l, *, target, random_state=0):
        assert len(train_t) == len(dev_tickets)
        assert len(test_t) == len(holdout_tickets)
        predictions = []
        for ticket in sorted(test_t, key=lambda t: t["id"]):
            true = gold[ticket["id"]][target]
            predicted = true
            if target == "issue_type" and ticket["id"] == "h06":
                predicted = "billing"
            if target == "product_area" and ticket["id"] == "h12":
                predicted = "authentication"
            predictions.append(baseline.Prediction(ticket["id"], true, predicted))
        return object(), predictions, 12.5

    monkeypatch.setattr(baseline, "train_and_predict", fake_train_and_predict)

    llm_report = tmp_path / "reports" / "holdout_report.json"
    llm_report.parent.mkdir(parents=True)
    llm_report.write_text(
        json.dumps(
            {
                "split": "holdout",
                "accuracy": {
                    "issue_type": {"accuracy_pct": 81.2},
                    "product_area": {"accuracy_pct": 87.5},
                },
            }
        ),
        encoding="utf-8",
    )

    report = baseline.run_baseline()

    assert report["train_split"] == "dev"
    assert report["test_split"] == "holdout"
    assert report["train_size"] == 40
    assert report["test_size"] == 16
    assert report["llm_report_used"] == "reports/holdout_report.json"
    assert report["targets"]["issue_type"]["accuracy"]["correct"] == 15
    assert report["targets"]["issue_type"]["delta_pct_points"] == 12.6
    assert report["targets"]["product_area"]["delta_pct_points"] == 6.3
    for target in ("issue_type", "product_area"):
        scores = report["targets"][target]
        assert set(scores["per_class"]) == set(scores["confusion_matrix"]["labels"])
        assert scores["macro_f1"] >= 0
        assert scores["inference_ms"]["total"] == 12.5


def test_main_writes_baseline_report(monkeypatch, tmp_path):
    _require_sklearn()
    output = tmp_path / "baseline.json"

    monkeypatch.setattr(
        baseline,
        "run_baseline",
        lambda **kwargs: {
            "train_split": "dev",
            "test_split": "holdout",
            "train_size": 40,
            "test_size": 16,
            "model": "fake",
            "random_state": 0,
            "targets": {
                "issue_type": {
                    "accuracy": {"correct": 13, "total": 16, "accuracy_pct": 81.2},
                    "macro_f1": 0.7,
                    "weighted_f1": 0.8,
                    "inference_ms": {"total": 1.0, "per_ticket": 0.062},
                    "llm_accuracy_pct": 81.2,
                    "delta_pct_points": 0.0,
                    "mismatches": [],
                }
            },
            "train_label_counts": {},
            "test_label_counts": {},
            "llm_report_used": None,
            "notes": [],
        },
    )

    baseline.main(["--targets", "issue_type", "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["targets"]["issue_type"]["accuracy"]["correct"] == 13
