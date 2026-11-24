from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .escalation import check_escalation
from .llm_client import Extractor, LLMExtractor
from .schema_validation import validate_extraction


@dataclass(frozen=True)
class TriageResult:
    id: str
    language: str
    issue_type: str
    priority: str
    product_area: str
    summary: str
    escalation_required: bool
    escalation_rule: str | None
    escalation_evidence: str | None
    send_to_oncall: bool
    prompt_tokens: int
    completion_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TriagePipeline:
    """Deterministic escalation first, schema-valid LLM extraction second."""

    def __init__(
        self,
        extractor: Extractor | None = None,
        *,
        max_validation_retries: int = 2,
    ) -> None:
        self.extractor = extractor or LLMExtractor()
        self.max_validation_retries = max_validation_retries

    def _extract_validated(
        self, subject: str, body: str
    ) -> tuple[dict[str, Any], int, int]:
        feedback: str | None = None
        prompt_tokens = 0
        completion_tokens = 0

        for attempt in range(self.max_validation_retries + 1):
            extraction = self.extractor.extract(subject, body, feedback=feedback)
            prompt_tokens += extraction.prompt_tokens
            completion_tokens += extraction.completion_tokens
            try:
                return (
                    validate_extraction(extraction.payload),
                    prompt_tokens,
                    completion_tokens,
                )
            except ValueError as error:
                if attempt == self.max_validation_retries:
                    raise
                feedback = str(error)

        raise AssertionError("unreachable")

    def triage(self, ticket: Mapping[str, Any]) -> TriageResult:
        subject = ticket["subject"]
        body = ticket["body"]

        escalation = check_escalation(f"{subject}\n{body}")
        fields, prompt_tokens, completion_tokens = self._extract_validated(
            subject, body
        )

        return TriageResult(
            id=ticket["id"],
            language=fields["language"],
            issue_type=fields["issue_type"],
            priority=fields["priority"],
            product_area=fields["product_area"],
            summary=fields["summary"],
            escalation_required=escalation.escalate,
            escalation_rule=escalation.rule,
            escalation_evidence=escalation.evidence,
            send_to_oncall=escalation.escalate,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
