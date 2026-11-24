# Nimbus support triage policy

Nimbus is a fictional SaaS product in this portfolio project: a shared workspace
where small teams keep documents, tasks, and customer invoices. The product, its
customers, and every ticket in `data/tickets.jsonl` were invented for this
project. No real customer data or third-party assignment material is used.

This file is the label contract given to the language model and used by the
evaluation. The model returns five fields; the deterministic escalation decision
is made separately by Python and is never produced by the model.

## Issue types

| Value | Use for |
| --- | --- |
| `login` | Cannot sign in, password reset, MFA, SSO, locked or disabled account |
| `billing` | Invoices, plans, seats, refunds, payment failures, taxes |
| `bug` | Something in the product is broken, slow, missing data, or shows wrong data |
| `feature_request` | A request for a new capability or integration |
| `other` | General questions, feedback, notices, or anything not covered above |

Disambiguation rules:

- Scope decides `login` vs `bug`: one account with a credential or access
  problem is `login`. The same symptom reported for the whole team, all users,
  a whole office/site, or the service itself is an active `bug`/incident, even
  when the visible symptom is "cannot sign in".
- Security and data-integrity incidents are `bug`: seeing another customer's
  data, a wrong-account view, or a public/leaked credential is a product fault.
  These are not `other` and not a billing question, even when invoices or
  passwords are the data involved.
- A concrete invoice, charge, plan, or payment problem is `billing` even when
  phrased as a question: wrong amount, unexpected tax, duplicate charge,
  missing invoice, or "something looks off". Pure how-to/configuration
  questions are `other` even when the topic is billing (for example, where to
  update the billing email, payment card, or invoice address); use
  `product_area` to capture the topic.
- Asking for a new field, integration, or behaviour is `feature_request`;
  asking how to use an existing setting is `other`.
- A follow-up that only confirms an issue is resolved and needs no action is
  `other`, not `bug`.

## Product areas

| Value | Use for |
| --- | --- |
| `authentication` | Sign-in, password reset, MFA, SSO |
| `billing` | Invoices, payments, plans, seats |
| `notifications` | Email, in-app, or push notifications |
| `integrations` | Slack, calendars, CRM connectors, Zapier-style tools |
| `mobile_app` | iOS and Android apps |
| `web_app` | Browser interface, exports, dashboards, uploads |
| `api` | REST API, tokens, webhooks, developer tooling |
| `admin_console` | Workspaces, members, permissions, space switching |
| `general` | No clear product area |

Choosing an area:

- Use the primary subject of the new message, not an incidental word. "Nobody
  can open/access Nimbus" is `web_app`. A problem specifically with credentials,
  MFA, or sign-in for one account/workspace is `authentication`. A service-wide
  or multi-user access/sign-in incident is `web_app` (and `issue_type` `bug`).
- Data visibility, workspace switching, guests, roles, and permissions are
  `admin_console`, regardless of which object happened to be visible.
- API keys, access tokens, webhooks, and REST/developer issues are `api`.
- Plan limits, quotas, storage allowances, invoices, seats, payments, and
  billing-account configuration topics are `billing`, including how-to questions
  labeled `other`.
- Custom fields, exports, dashboards, uploads, and the browser interface are
  `web_app`; the panel/dashboard is `web_app` even for a vague report.
- Named third-party connectors and calendars are `integrations`; mobile apps
  are `mobile_app`; email and in-app notifications are `notifications`.

## Priority

| Value | Meaning |
| --- | --- |
| `urgent` | An active incident affecting many users or exposing data |
| `normal` | Something broken or a concrete billing/account problem needing action |
| `low` | Questions, feature requests, feedback, resolved issues |

Priority is independent of the escalation decision. The Python pre-check may
send a ticket to the on-call engineer even when the model returns `normal`.

- Active data exposure, cross-customer visibility, or a public credential is
  `urgent`, even if the customer asks politely or says the item was deleted.
- A wrong amount, unexpected tax, duplicate charge, or unexplained change on an
  invoice is a concrete billing problem (`normal`), not `low`. `low` is for
  how-to questions, preferences, feature requests, and resolved issues.
- A service-wide outage is `urgent`; a single-user fault is `normal`.

## Language

One of `en`, `de`, `fr`, `es`. Classify the language of the ticket text itself,
not the sender's email domain or a language mentioned in the ticket.

## Tie-breakers

- Intent beats topic for `issue_type`: classify what the user needs (how-to,
  broken behavior, concrete billing problem, new capability), then use
  `product_area` for the product/topic named. A how-to question about invoices
  or payment settings is `issue_type` `other`, `product_area` `billing`.
- Scope beats symptom for access failures: if multiple users, a whole team,
  office, or "everyone" is affected, treat it as a service/product incident
  (`issue_type` `bug`, `product_area` `web_app`, usually `urgent`) rather than
  `login`/`authentication`.
- Classify the primary complaint in the subject/first sentence of the new
  message. Later restatements or secondary symptoms do not change the label.
- Security/permission incidents override topic-based area rules. If the report
  is about seeing data you should not see, the area is `admin_console` even
  when the exposed objects are invoices.
- Reserve `general` for tickets that name no product surface at all. A vague
  report that mentions a panel, dashboard, page, or screen belongs to `web_app`.

## Deterministic escalation (Python, not the model)

Three predefined cases always go to a human on-call engineer:

1. `service_wide_outage` - the service is reported as down for everyone, the
   whole team, or all users. A single user who cannot sign in is not an outage.
2. `data_exposure` - a user can see another company's or customer's data.
3. `credential_leak` - an API key, token, or password is public, leaked, or
   committed to a public repository.

The pre-check only reads the new message. Quoted history (lines starting with
`>` or following a "wrote/schrieb/écrit/escribió:" header) is ignored, so an old
outage in a forwarded thread does not re-escalate. Negated and resolved reports
("the service is not down", "pas de panne", "ya está resuelto") do not escalate.
The rules are intentionally simple and prefer false positives over missing a
real incident; they are documented, testable, and cannot be overridden by the
model or by instructions embedded in a ticket.
