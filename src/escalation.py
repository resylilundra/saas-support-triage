from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class EscalationRule:
    name: str
    patterns: tuple[str, ...]


RULES: tuple[EscalationRule, ...] = (
    EscalationRule(
        "service_wide_outage",
        (
            r"\boutage\b",
            r"\b(?:service|platform|app|api|nimbus) down "
            r"(?:for everyone|for all|across|for the whole)\b",
            r"\b(?:service|platform|app|api|nimbus) is "
            r"(?:not |currently |completely |totally ){0,2}"
            r"(?:down|unavailable|offline)\b",
            r"\b(?:platform|app|api|nimbus) (?:is|has been|was) "
            r"(?:down|unavailable|offline)\b",
            r"\b(?:whole|our whole|the whole) (?:team|office|company) "
            r"(?:has been|have been|is|are) unable to\b",
            r"\bnobody (?:in|at) (?:the|our) (?:office|team|company) "
            r"can (?:sign in|log in|access|open)\b",
            r"\ball (?:of )?(?:our|the) users\b[^.!?]{0,60}"
            r"\b(?:blocked|unable|cannot|can not)\b",
            r"\balle\b[^.!?]{0,60}\b(?:nicht|keine)\b[^.!?]{0,20}"
            r"\b(?:anmelden|einloggen|zugreifen|offnen)\b",
            r"\b(?:gerade|jetzt|aktuell) (?:einen )?ausfall\b",
            r"\bausfall\b",
            r"\btoute l'equipe\b[^.!?]{0,60}\b(?:ne peut plus|n'arrive pas|peut pas)\b"
            r"[^.!?]{0,30}\b(?:acceder|se connecter|ouvrir)\b",
            r"\bpersonne ne peut (?:se connecter|acceder|ouvrir)\b",
            r"\bpanne (?:en cours|generale|generalisee)\b",
            r"\bpanne\b",
            r"\btodo el equipo\b[^.!?]{0,60}\b(?:sin poder|no puede|no pueden|no podemos)\b",
            r"\bnadie puede (?:iniciar sesion|acceder|entrar)\b",
            r"\bcaida (?:del servicio|en curso|general)\b",
            r"\bcaida\b",
        ),
    ),
    EscalationRule(
        "data_exposure",
        (
            r"\banother (?:company|customer|user|team|client)'?s? "
            r"(?:data|invoice|invoices|file|files|document|documents)\b",
            r"\b(?:i|we) can see another (?:company|customer|user|team)\b",
            r"\b(?:someone else|another user) (?:can|could) "
            r"(?:see|access|read) (?:my|our)\b",
            r"\b(?:fremde|andere) (?:daten|rechnungen|dateien|dokumente) "
            r"(?:sichtbar|zu sehen|gesehen)\b",
            r"\b(?:daten|rechnungen|dateien) (?:einer anderen firma|anderer kunden|"
            r"eines anderen kunden)\b",
            r"\b(?:ein anderer|jemand anderes) (?:kann|konnte) (?:unsere|meine) "
            r"(?:daten|rechnungen|dateien) (?:sehen|zugreifen)\b",
            r"\bdonnees (?:d'une autre entreprise|d'un autre client|d'autres clients|"
            r"d'un autre compte)\b",
            r"\b(?:un autre utilisateur|quelqu'un d'autre) (?:peut|pouvait) "
            r"(?:voir|acceder a) (?:mes|nos) (?:donnees|factures)\b",
            r"\bdatos (?:de otra empresa|de otro cliente|de otros clientes|"
            r"de otra cuenta)\b",
            r"\b(?:otro usuario|alguien mas) (?:puede|podia) (?:ver|acceder a) "
            r"(?:mis|nuestros) (?:datos|facturas)\b",
        ),
    ),
    EscalationRule(
        "credential_leak",
        (
            r"\b(?:api key|access token|password) (?:is|was|got) "
            r"(?:public|exposed|leaked|committed)\b",
            r"\b(?:api key|access token|password)\b[^.!?]{0,60}\b(?:public|exposed|leaked)\b"
            r"[^.!?]{0,40}\b(?:repo|repository|github|gitlab)\b",
            r"\b(?:leaked|exposed) (?:api key|credentials|token|password)\b",
            r"\b(?:api-schlussel|api key|zugriffstoken|passwort)\b[^.!?]{0,70}"
            r"\b(?:offentlich|geleakt|veroeffentlicht|public)\b",
            r"\boffentlich(?:es|en|e|er)?\b[^.!?]{0,40}\b(?:repo|repository|github|gitlab)\b",
            r"\b(?:jeton|cle api|mot de passe)\b[^.!?]{0,70}\b(?:public|expose|fuite)\b",
            r"\b(?:contrasena|clave api|token)\b[^.!?]{0,70}"
            r"\b(?:publico|publica|expuesta|filtrada)\b",
        ),
    ),
)

QUOTED_LINE = re.compile(r"^\s*>")
QUOTE_HEADER = re.compile(
    r"^\s*(?:on|am|le|el|il)\b.{0,100}\b"
    r"(?:wrote|schrieb|ecrit|escribio|scritto)\s*:\s*$",
    re.IGNORECASE,
)
CLAUSE_BOUNDARY = re.compile(
    r"[.!?;:,\n]|\b(?:but|however|although|though|aber|jedoch|mais|"
    r"cependant|pero|tuttavia)\b"
)
NEGATIONS = re.compile(
    r"\b(?:no|not|never|without|none|nobody|nothing)\b"
    r"|n't\b"
    r"|\b(?:kein|keine|keinen|keinem|keiner|nicht|ohne|nie|niemals)\b"
    r"|\b(?:pas|aucun|aucune|sans|jamais|rien)\b"
    r"|\b(?:ningun|ninguna|ninguno|sin|nunca|nada)\b"
)
RESOLVED = re.compile(
    r"\b(?:fixed|resolved|restored|back online|works again|working again|"
    r"no action needed|behoben|gelost|wieder online|wieder verfugbar|"
    r"resolu|regle|retabli|resuelto|arreglado|ya no)\b"
)
SINGLE_CUSTOMER = re.compile(
    r"\b(?:one|single|only) (?:customer|user|account|workspace|phone|device)\b"
    r"|\b(?:nur (?:unser|mein) (?:account|konto)|seul compte|un solo usuario)\b"
)
EMBEDDED_NEGATION_TOKENS = (
    "nobody",
    "nicht",
    "keine",
    "sin",
    "no puede",
    "no pueden",
    "no podemos",
)


@dataclass(frozen=True)
class EscalationResult:
    escalate: bool
    rule: str | None
    evidence: str | None


def normalize(text: str) -> str:
    """Casefold and strip accents so rules work across the four languages."""
    text = text.casefold()
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def strip_quoted_history(text: str) -> str:
    """Remove quoted lines and forwarded-thread headers from a ticket."""
    kept: list[str] = []
    in_quoted_block = False
    for line in text.splitlines():
        if QUOTED_LINE.match(line):
            continue
        if QUOTE_HEADER.match(line):
            in_quoted_block = True
            continue
        if in_quoted_block:
            continue
        kept.append(line)
    return "\n".join(kept)


def _clause_window(text: str, start: int, end: int) -> str:
    before = text[max(0, start - 100) : start]
    last_boundary = None
    for boundary in CLAUSE_BOUNDARY.finditer(before):
        last_boundary = boundary
    if last_boundary is not None:
        before = before[last_boundary.end() :]

    after = text[end : end + 60]
    first_boundary = CLAUSE_BOUNDARY.search(after)
    if first_boundary is not None:
        after = after[: first_boundary.start()]

    return f"{before}{text[start:end]}{after}"


def check_escalation(text: str) -> EscalationResult:
    """Apply deterministic rules to the new, unquoted part of a ticket."""
    visible_text = normalize(strip_quoted_history(text))

    for rule in RULES:
        for pattern in rule.patterns:
            for match in re.finditer(pattern, visible_text):
                clause = _clause_window(visible_text, match.start(), match.end())
                negation_scope = clause
                if any(token in pattern for token in EMBEDDED_NEGATION_TOKENS):
                    negation_scope = clause.replace(match.group(0), " ")
                single_customer = rule.name == "service_wide_outage" and bool(
                    SINGLE_CUSTOMER.search(clause)
                )
                if (
                    NEGATIONS.search(negation_scope)
                    or RESOLVED.search(clause)
                    or single_customer
                ):
                    continue
                return EscalationResult(True, rule.name, match.group(0))

    return EscalationResult(False, None, None)
