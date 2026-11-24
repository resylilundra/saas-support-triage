import pytest

from src.escalation import check_escalation, strip_quoted_history


@pytest.mark.parametrize(
    ("text", "expected_rule"),
    [
        ("Our whole team has been unable to open Nimbus since 09:00.", "service_wide_outage"),
        ("Nobody in the office can sign in right now.", "service_wide_outage"),
        ("Is there an outage in progress?", "service_wide_outage"),
        ("All of our users who create tasks are blocked.", "service_wide_outage"),
        ("Seit 8 Uhr können sich bei uns alle nicht mehr anmelden.", "service_wide_outage"),
        ("Gibt es gerade einen Ausfall?", "service_wide_outage"),
        ("Toute l'équipe ne peut plus accéder à Nimbus.", "service_wide_outage"),
        ("Est-ce une panne en cours ?", "service_wide_outage"),
        ("Todo el equipo lleva desde las 9 sin poder acceder.", "service_wide_outage"),
        ("¿Hay una caída del servicio en curso?", "service_wide_outage"),
        ("I can see another company's invoices.", "data_exposure"),
        ("Ich konnte die Rechnungen einer anderen Firma sehen.", "data_exposure"),
        ("Je vois les données d'une autre entreprise.", "data_exposure"),
        ("Veo los datos de otra empresa en la lista.", "data_exposure"),
        ("We committed an API key to a public repository.", "credential_leak"),
        ("Ein API-Schlüssel steht in einem öffentlichen GitHub-Repo.", "credential_leak"),
        ("Un jeton d'accès est dans un canal Slack public.", "credential_leak"),
        ("Subimos una contraseña a un repositorio público.", "credential_leak"),
    ],
)
def test_predefined_incidents_escalate(text, expected_rule):
    result = check_escalation(text)

    assert result.escalate is True
    assert result.rule == expected_rule
    assert result.evidence


@pytest.mark.parametrize(
    "text",
    [
        "I can't sign in after resetting my password; I'm the only one affected.",
        "Since this morning no emails arrive, but my colleagues get theirs.",
        "Could you add a Slack integration?",
        "The service is not down, only the Android app on one phone.",
        "The service is currently down for one customer only.",
        "Es gibt keinen Ausfall, nur mein Handy.",
        "Il n'y a pas de panne chez vous.",
        "No hay caída del servicio, es mi teléfono.",
        "Last week's outage is fixed, thanks.",
        "The export works again now; that issue is resolved.",
        "Where can I see how much storage our plan includes?",
    ],
)
def test_non_incidents_do_not_escalate(text):
    assert check_escalation(text).escalate is False


def test_quoted_history_does_not_escalate():
    text = (
        "Thanks for the fix, the export works again now.\n\n"
        "> On Monday the service was down for everyone.\n"
        "> Please treat this as urgent.\n\n"
        "No action needed.\n"
    )

    result = check_escalation(text)

    assert result.escalate is False
    assert "> On Monday" not in strip_quoted_history(text)


def test_forwarded_thread_header_removes_following_lines():
    text = (
        "Please check the new invoice.\n\n"
        "On 3 November Clara wrote:\n> The service is down for everyone."
    )

    result = check_escalation(text)

    assert result.escalate is False


def test_unrelated_negation_does_not_hide_real_incident():
    result = check_escalation(
        "I don't know who to call, but the service is down for the whole team."
    )

    assert result.escalate is True
