# Multilingual SaaS Support Triage

A clean-room portfolio project: triage Nimbus support tickets (English, German, French, Spanish) with deterministic safety rules + schema-validated LLM extraction, plus evaluation and a scikit-learn baseline.

**Nimbus** and all tickets/labels are fictional and synthetic for this project (no real customer or third-party assignment data).

## How it works

1. **Preprocess:** strip quoted/forwarded history so old incidents cannot re-trigger escalation.
2. **Deterministic escalation (Python, not the LLM):** named regex rules detect `service_wide_outage`, `data_exposure`, and `credential_leak`. They set `escalation_required`, `escalation_rule`, `escalation_evidence`, and `send_to_oncall`. The model cannot override routing (prompt-injection resistant for escalation).
3. **LLM extraction:** model returns 5 strict fields — `language`, `issue_type`, `priority`, `product_area`, `summary` (JSON Schema, `additionalProperties: false`, enums). Invalid output triggers a bounded retry with validation feedback; persistent failure raises.
4. **Evaluation:** scores field accuracy/mismatches, latency, tokens, and estimated API cost against gold labels (`summary` not scored).

Label/taxonomy rules live in `data/triage_policy.md`; schema in `data/schema.json`.

## Project layout

```text
data/
  triage_policy.md, schema.json
  tickets.jsonl / gold_labels.jsonl             # dev: 40 (10/lang), used for tuning
  holdout_tickets.jsonl / holdout_labels.jsonl  # holdout: 16 (4/lang), frozen
  train_tickets.jsonl / train_labels.jsonl      # train: 360, synthetic/balanced
  test_tickets.jsonl / test_labels.jsonl        # test: 80, synthetic/balanced
src/
  escalation.py        # quote strip, negation/resolved handling, rules
  llm_client.py        # injectable extractor (chat completions)
  schema_validation.py
  triage.py            # pipeline + retry
  evaluate.py          # LLM evaluation/scoring
  baseline.py          # sklearn baseline (side-by-side, not in pipeline)
scripts/
  generate_large_dataset.py
tests/                 # offline, no API calls
reports/               # generated JSON reports (gitignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # pytest, jsonschema, openai, scikit-learn
cp .env.example .env                     # set OPENAI_API_KEY=...
set -a; source .env; set +a
```

`OPENAI_API_KEY` is needed only for live LLM evaluation; tests and the sklearn baseline do not use it.

## Run (end-to-end)

```bash
# 1) Offline checks
python -m pytest -q

# 2) (Optional) regenerate expanded synthetic train/test data
python scripts/generate_large_dataset.py

# 3) LLM evaluation (1 API call per ticket; costs tokens)
python -m src.evaluate --split dev --limit 4   # smoke test
python -m src.evaluate --split dev            # -> reports/dev_report.json
python -m src.evaluate --split holdout        # -> reports/holdout_report.json
python -m src.evaluate --split test           # -> reports/test_report.json (80 calls)

# 4) Scikit-learn baseline (local, no API/cost), predicts issue_type & product_area
python -m src.baseline                        # train->test, -> reports/baseline_test_report.json
python -m src.baseline --train-split dev --test-split holdout  # illustrative small-data compare
python -m src.baseline --llm-report reports/test_report.json   # delta vs LLM on test split
```

`src.evaluate` options: `--model gpt-4o-mini` (default), `--output <path>`. Pricing constants are in `src/evaluate.py` (update before reporting cost). Reports are printed and saved as JSON.

## Data splits

| Split | Size | Purpose |
|---|---:|---|
| `dev` | 40 (10/lang) | Tuning/prompt iteration (**in-sample**) |
| `holdout` | 16 (4/lang) | Written after prompt freeze; untouched generalization check |
| `train` | 360 (90/lang) | Synthetic, balanced (18/class/lang) for sklearn baseline |
| `test` | 80 (20/lang) | Synthetic, balanced (4/class/lang) for sklearn baseline |

`train`/`test` are template-generated from the clarified policy with escalation labels verified against `src/escalation.py`. They are disjoint from dev/holdout and from each other, and use disjoint template texts (no exact/normalized template leakage) to avoid inflating baseline scores. Regenerate with `python scripts/generate_large_dataset.py`. **Do not tune on `holdout` or `test`;** if you change the system after seeing them, treat that split as burned and create a fresh one.

## Results (from generated reports, `gpt-4o-mini`)

| Field | Dev 40 (in-sample) | Holdout 16 (untouched) | Test 80 (synthetic, prior run) |
|---|---:|---:|---:|
| Language | 100.0% | 100.0% | 100.0% |
| Issue type | 97.5% (39/40) | 100.0% | 100.0% |
| Priority | 100.0% | 100.0% | 100.0% |
| Product area | 95.0% (38/40) | 100.0% | 98.8% (79/80) |
| Escalation | 100.0% | 100.0% | 100.0% |

- **Dev mismatches:** t06 `issue_type bug!=login`, `product_area web_app!=authentication`; t40 `product_area general!=web_app`.
- **Test (prior run):** `reports/test_report.json` reflects an earlier synthetic test set (before fixing train/test template leakage); it includes fr-test-014 `product_area general!=web_app` and ~$0.0258 cost. After regenerating disjoint train/test templates the LLM test split was **not re-evaluated** here (requires API key/calls); re-run `python -m src.evaluate --split test` to refresh.
- **Cost/latency (approx., dev/holdout):** dev ~$0.0130 (78.6k in / 2.1k out tokens), holdout ~$0.0052.
- **Interpretation:** deterministic escalation stayed 100% (safety layer works independently of the LLM). Dev is in-sample and was improved through iteration/taxonomy clarification (earlier runs were lower before fixing ambiguities like how-to-vs-billing intent and multi-user access incidents vs login); holdout reflects the clarified policy on untouched data. Treat results as local runs, not benchmarks.

## Scikit-learn baseline

- **What:** `TfidfVectorizer(analyzer="char_wb", ngram_range=(1,2)) + LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)` predicting `issue_type` and `product_area`, evaluated side-by-side (accuracy, macro/weighted F1, per-class precision/recall, confusion matrix, inference time). **Not** wired into `src/triage.py` and no API cost.
- **Current results (regenerated, disjoint train/test templates):** train 360 -> test 80 gives **issue_type 83.8% (67/80, macro F1 0.827)** and **product_area 76.2% (61/80, macro F1 0.725)** at ~0.2 ms/ticket. Dev 40 -> holdout 16 (small/imbalanced, illustrative): 43.8% on both targets (macro F1 ~0.31–0.35) vs LLM holdout 100%.
- **Caveat:** train/test are synthetic and template-structured (now without template leakage), so scores do not represent real-ticket performance and labels are not human-reviewed. The dev/holdout comparison is limited by data size/coverage. A strong “LLM vs supervised” claim needs a few hundred real, reviewed tickets and a fresh untouched test set.

## Design decisions & limitations

- **Safety first:** escalation/routing is deterministic, tested, and non-overridable; rules are biased toward safety on incidents.
- **Robust input handling:** quoted/forwarded history removed; negated/resolved and single-customer cases suppressed to reduce false escalations (regex can still miss paraphrases or over-trigger).
- **Strict contract:** JSON Schema boundary + bounded validation retries with token accounting; policy lives in data.
- **Limits:** summaries unscored, simple bounded retries (no cache/confidence/fallback), accent/case normalization is not full language handling, cost/latency are single-run measurements, no PII/real feed/persistence.
