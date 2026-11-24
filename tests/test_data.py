from collections import Counter

import pytest
from jsonschema import validate

from src.escalation import check_escalation
from src.evaluate import load_jsonl
from src.schema_validation import SCHEMA


@pytest.fixture(scope="module")
def tickets():
    from src.evaluate import TICKETS_PATH

    return load_jsonl(TICKETS_PATH)


@pytest.fixture(scope="module")
def labels():
    from src.evaluate import LABELS_PATH

    return load_jsonl(LABELS_PATH)


@pytest.fixture(scope="module")
def holdout_tickets():
    from src.evaluate import HOLDOUT_TICKETS_PATH

    return load_jsonl(HOLDOUT_TICKETS_PATH)


@pytest.fixture(scope="module")
def holdout_labels():
    from src.evaluate import HOLDOUT_LABELS_PATH

    return load_jsonl(HOLDOUT_LABELS_PATH)


def test_dataset_size_and_unique_ids(tickets, labels):
    assert len(tickets) == 40
    assert len(labels) == 40
    assert len({ticket["id"] for ticket in tickets}) == 40
    assert {ticket["id"] for ticket in tickets} == {label["id"] for label in labels}


def test_dataset_is_multilingual_and_balanced(labels):
    language_counts = Counter(label["language"] for label in labels)
    assert language_counts == {"en": 10, "de": 10, "fr": 10, "es": 10}


def test_gold_labels_use_schema_enums(labels):
    for label in labels:
        validate(
            instance={
                key: label.get(key, "Generated summary.") for key in SCHEMA["required"]
            },
            schema=SCHEMA,
        )


def test_deterministic_rules_match_gold_labels(tickets, labels):
    labels_by_id = {label["id"]: label for label in labels}
    for ticket in tickets:
        result = check_escalation(f"{ticket['subject']}\n{ticket['body']}")
        label = labels_by_id[ticket["id"]]
        assert result.escalate is label["escalation_required"], ticket["id"]
        assert result.rule == label["escalation_rule"], ticket["id"]


def test_resolved_follow_up_is_labeled_as_other(labels):
    label = next(label for label in labels if label["id"] == "t17")

    assert label["issue_type"] == "other"
    assert label["escalation_required"] is False


def test_holdout_is_separate_and_balanced(tickets, holdout_tickets, holdout_labels):
    assert len(holdout_tickets) == 16
    assert len(holdout_labels) == 16
    assert len({ticket["id"] for ticket in holdout_tickets}) == 16
    assert {ticket["id"] for ticket in holdout_tickets} == {
        label["id"] for label in holdout_labels
    }
    dev_ids = {ticket["id"] for ticket in tickets}
    holdout_ids = {ticket["id"] for ticket in holdout_tickets}
    assert not (dev_ids & holdout_ids)
    assert Counter(label["language"] for label in holdout_labels) == {
        "en": 4,
        "de": 4,
        "fr": 4,
        "es": 4,
    }


def test_holdout_labels_and_rules_are_consistent(holdout_tickets, holdout_labels):
    labels_by_id = {label["id"]: label for label in holdout_labels}
    for label in holdout_labels:
        validate(
            instance={
                key: label.get(key, "Generated summary.") for key in SCHEMA["required"]
            },
            schema=SCHEMA,
        )
    for ticket in holdout_tickets:
        result = check_escalation(f"{ticket['subject']}\n{ticket['body']}")
        label = labels_by_id[ticket["id"]]
        assert result.escalate is label["escalation_required"], ticket["id"]
        assert result.rule == label["escalation_rule"], ticket["id"]


def test_dataset_includes_adversarial_cases(tickets):
    by_id = {ticket["id"]: ticket for ticket in tickets}

    assert (
        "> On Monday the service was down for everyone and exports were failing."
        in by_id["t17"]["body"]
    )
    assert "Please treat this as urgent." in by_id["t17"]["body"]
    assert "the service is not down" in by_id["t25"]["body"]
    assert "no hay caída del servicio" in by_id["t28"]["body"]
    assert "Can you clarify?" in by_id["t37"]["subject"]

@pytest.fixture(scope="module")
def train_tickets():
    from src.evaluate import TRAIN_TICKETS_PATH

    return load_jsonl(TRAIN_TICKETS_PATH)


@pytest.fixture(scope="module")
def train_labels():
    from src.evaluate import TRAIN_LABELS_PATH

    return load_jsonl(TRAIN_LABELS_PATH)


@pytest.fixture(scope="module")
def test_tickets():
    from src.evaluate import TEST_TICKETS_PATH

    return load_jsonl(TEST_TICKETS_PATH)


@pytest.fixture(scope="module")
def test_labels():
    from src.evaluate import TEST_LABELS_PATH

    return load_jsonl(TEST_LABELS_PATH)


def test_large_train_test_splits_exist_and_are_balanced_by_language(
    train_tickets,
    train_labels,
    test_tickets,
    test_labels,
    tickets,
    holdout_tickets,
):
    assert len(train_tickets) == 360
    assert len(train_labels) == 360
    assert len(test_tickets) == 80
    assert len(test_labels) == 80
    assert {t["id"] for t in train_tickets} == {l["id"] for l in train_labels}
    assert {t["id"] for t in test_tickets} == {l["id"] for l in test_labels}
    assert not ({t["id"] for t in train_tickets} & {t["id"] for t in test_tickets})
    # expanded splits are separate from the original dev/holdout evaluation sets
    assert not ({t["id"] for t in train_tickets} & {t["id"] for t in tickets})
    assert not ({t["id"] for t in test_tickets} & {t["id"] for t in holdout_tickets})
    assert Counter(l["language"] for l in train_labels) == {"en": 90, "de": 90, "fr": 90, "es": 90}
    assert Counter(l["language"] for l in test_labels) == {"en": 20, "de": 20, "fr": 20, "es": 20}


def test_large_split_labels_and_rules_are_consistent(train_tickets, train_labels, test_tickets, test_labels):
    for split_tickets, split_labels in [
        (train_tickets, train_labels),
        (test_tickets, test_labels),
    ]:
        labels_by_id = {label["id"]: label for label in split_labels}
        for label in split_labels:
            validate(
                instance={
                    key: label.get(key, "Generated summary.") for key in SCHEMA["required"]
                },
                schema=SCHEMA,
            )
        for ticket in split_tickets:
            result = check_escalation(f"{ticket['subject']}\n{ticket['body']}")
            label = labels_by_id[ticket["id"]]
            assert result.escalate is label["escalation_required"], ticket["id"]
            assert result.rule == label["escalation_rule"], ticket["id"]


def test_large_splits_have_no_template_overlap(train_tickets, test_tickets):
    def norm(ticket):
        lines = ticket["body"].rstrip().split("\n")
        if lines:
            lines[-1] = "<NAME>"
        return (ticket["subject"], "\n".join(lines))

    train_exact = {(t["subject"], t["body"]) for t in train_tickets}
    test_exact = {(t["subject"], t["body"]) for t in test_tickets}
    assert not (train_exact & test_exact)

    train_norm = {norm(t) for t in train_tickets}
    test_norm = {norm(t) for t in test_tickets}
    assert not (train_norm & test_norm)
