from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RANDOM_SEED = 20261119

# Balanced synthetic dataset for sklearn baseline experiments.
# Per-language: 90 train + 20 test, 18/4 per issue_type.
# Train and test use disjoint template texts to avoid leakage.
TRAIN_PER_TYPE = 18
TEST_PER_TYPE = 4
ISSUE_TYPES = ("login", "billing", "bug", "feature_request", "other")
TRAIN_ESCALATION_PATTERN = [
    "service_wide_outage",
    "service_wide_outage",
    "data_exposure",
    "credential_leak",
]
TEST_ESCALATION_PATTERN = [
    "service_wide_outage",
    "data_exposure",
    "credential_leak",
    None,
]


def grouped(cases):
    by_type = defaultdict(list)
    for case in cases:
        by_type[case[2]].append(case)
    return by_type


def cases_en():
    train = grouped([
        ("Can't sign in after reset", "Hi,\n\nI reset my password but Nimbus still says it is incorrect. I'm the only one affected.\n\nThanks,\n{name}", "login", "normal", "authentication", False, None),
        ("MFA code never arrives", "Hello,\n\nI am trying to log in and the MFA code never reaches my phone. Nobody else on my team reported this.\n\n{name}", "login", "normal", "authentication", False, None),
        ("SSO login fails for my account", "Hi,\n\nSSO redirects me back to the sign-in page for my account only. Could you check what is wrong?\n\n{name}", "login", "normal", "authentication", False, None),

        ("Invoice amount looks wrong", "Hi,\n\nOur latest invoice includes a charge we do not recognize. Could you review the amount?\n\n{name}", "billing", "normal", "billing", False, None),
        ("Duplicate charge on invoice", "Hello,\n\nWe were charged twice on this month's invoice. Please investigate and correct it.\n\n{name}", "billing", "normal", "billing", False, None),
        ("Missing invoice for last month", "Hi,\n\nI cannot find last month's invoice in the billing section. Can you help locate it?\n\n{name}", "billing", "normal", "billing", False, None),

        ("Export fails in browser", "Hi,\n\nwhen I export a document from the web app, it fails with an error. My colleagues can reproduce it on the same file.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Dashboard shows wrong totals", "Hello,\n\nthe dashboard totals are incorrect for our workspace today. Numbers don't match the task list.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Upload stuck at 99%", "Hi,\n\nfile uploads get stuck at 99% in the browser and never complete.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Mobile app crashes on open", "Hello,\n\nthe Nimbus mobile app crashes immediately after I open it on my phone.\n\n{name}", "bug", "normal", "mobile_app", False, None),
        ("Notifications not delivered", "Hi,\n\nI stopped receiving email notifications for task updates since yesterday.\n\n{name}", "bug", "normal", "notifications", False, None),
        ("Slack integration stopped syncing", "Hello,\n\nour Slack integration no longer posts updates to the channel. It worked last week.\n\n{name}", "bug", "normal", "integrations", False, None),
        ("API webhook returns 500", "Hi,\n\nour webhook endpoint receives 500 errors from the Nimbus API for document events.\n\n{name}", "bug", "normal", "api", False, None),
        ("Guest permissions not saving", "Hello,\n\nchanges to guest permissions in the admin console are not saved after I click confirm.\n\n{name}", "bug", "normal", "admin_console", False, None),
        ("Nobody can access Nimbus", "Hello,\n\nnobody in the office can sign in to Nimbus right now, on desktop or mobile. Is there an outage?\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Service down for all users", "Hi,\n\nthe Nimbus service is down for everyone in our company right now. No one can open the app.\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("I can see another company's invoices", "Hi,\n\nI can see another company's invoices in the workspace switcher. Please check this urgently.\n\n{name}", "bug", "urgent", "admin_console", True, "data_exposure"),
        ("API key committed to public repo", "Hello,\n\nwe accidentally committed an API key to a public GitHub repository. Please advise immediately.\n\n{name}", "bug", "urgent", "api", True, "credential_leak"),

        ("Can you add calendar integration?", "Hi,\n\nit would be great if Nimbus could sync tasks with Google Calendar. Is that planned?\n\n{name}", "feature_request", "low", "integrations", False, None),
        ("Request for custom task fields", "Hello,\n\ncould you add custom fields to tasks? We need them for our workflow.\n\n{name}", "feature_request", "low", "web_app", False, None),
        ("Dark mode for mobile app", "Hi,\n\nplease consider adding dark mode to the mobile app.\n\n{name}", "feature_request", "low", "mobile_app", False, None),

        ("Where can I change the billing email?", "Quick question: where can I change the email address that receives invoices? I looked in Settings but couldn't find it.\n\n{name}", "other", "low", "billing", False, None),
        ("How do I update the payment card?", "Hi,\n\nhow do I update the card used for monthly payments? I can't find the option.\n\nThanks,\n{name}", "other", "low", "billing", False, None),
        ("How to download old invoices?", "Hello,\n\nwhere can I download invoices from previous months? I only see the current one.\n\n{name}", "other", "low", "billing", False, None),
        ("General question about plans", "Hi,\n\ncan you explain the difference between your workspace plans? I'm not sure which one fits us.\n\n{name}", "other", "low", "general", False, None),
        ("How to invite a guest?", "Hello,\n\nhow can I invite a guest to only one workspace? I couldn't find clear steps.\n\n{name}", "other", "low", "admin_console", False, None),
        ("How to use notification settings?", "Hi,\n\nwhere can I choose which task notifications I receive?\n\n{name}", "other", "low", "notifications", False, None),
    ])
    test = grouped([
        ("Password reset still not accepted", "Hi,\n\nAfter changing my password I still cannot sign in to Nimbus. This only happens on my account.\n\nRegards,\n{name}", "login", "normal", "authentication", False, None),
        ("No verification code received", "Hello,\n\nI never receive the MFA verification code when logging in from my laptop. No one else is affected.\n\n{name}", "login", "normal", "authentication", False, None),

        ("Charge amount on bill is unexpected", "Hi,\n\nThe latest bill includes an extra line item I don't recognize. Can someone review it?\n\n{name}", "billing", "normal", "billing", False, None),
        ("Billed twice this cycle", "Hello,\n\nIt looks like our workspace was charged twice this billing cycle. Please help correct this.\n\n{name}", "billing", "normal", "billing", False, None),

        ("Report export returns an error", "Hi,\n\nExporting a report in the web app returns an error for several people on our team.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Attachment upload never finishes", "Hello,\n\nUploading attachments in the browser never completes; it stays pending.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Nimbus outage for the whole office", "Hi,\n\nnobody in the office can sign in to Nimbus at the moment. The service is down for everyone here.\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Another company's invoices visible", "Hello,\n\nI can see another company's invoices when switching workspaces. This needs urgent review.\n\n{name}", "bug", "urgent", "admin_console", True, "data_exposure"),
        ("Access token exposed in public repository", "Hi,\n\nAn access token was committed to a public GitHub repository by mistake. Please advise on next steps.\n\n{name}", "bug", "urgent", "api", True, "credential_leak"),

        ("Request: task templates", "Hi,\n\nWould it be possible to add reusable task templates in a future release?\n\n{name}", "feature_request", "low", "web_app", False, None),
        ("Suggestion for audit log export", "Hello,\n\nIt would help if we could export audit logs from the admin console.\n\n{name}", "feature_request", "low", "admin_console", False, None),

        ("How can I change invoice recipient?", "Hi,\n\nWhere do I update which email address receives our invoices?\n\n{name}", "other", "low", "billing", False, None),
        ("Where to edit payment method?", "Hello,\n\nCan you tell me where to edit the payment method for our subscription?\n\n{name}", "other", "low", "billing", False, None),
    ])
    return train, test


def cases_de():
    train = grouped([
        ("Anmeldung nach Reset nicht möglich", "Guten Tag,\n\nich habe mein Passwort zurückgesetzt, kann mich aber weiterhin nicht anmelden. Nur mein Konto ist betroffen.\n\n{name}", "login", "normal", "authentication", False, None),
        ("MFA-Code kommt nicht an", "Hallo,\n\nbeim Login erhalte ich keinen MFA-Code auf meinem Telefon. Andere im Team haben das Problem nicht.\n\n{name}", "login", "normal", "authentication", False, None),
        ("SSO funktioniert nicht für mich", "Guten Tag,\n\nSSO leitet nur mein Konto immer zurück zur Anmeldung. Können Sie das prüfen?\n\n{name}", "login", "normal", "authentication", False, None),

        ("Rechnungsbetrag stimmt nicht", "Guten Tag,\n\nauf unserer letzten Rechnung ist ein unbekannter Betrag. Bitte prüfen Sie das.\n\n{name}", "billing", "normal", "billing", False, None),
        ("Doppelte Abbuchung", "Hallo,\n\nwir wurden diesen Monat doppelt belastet. Bitte korrigieren Sie die Rechnung.\n\n{name}", "billing", "normal", "billing", False, None),
        ("Rechnung vom letzten Monat fehlt", "Guten Tag,\n\nich finde die Rechnung vom letzten Monat nicht im Abrechnungsbereich.\n\n{name}", "billing", "normal", "billing", False, None),

        ("Export schlägt im Browser fehl", "Hallo,\n\nder Export eines Dokuments im Web-App schlägt fehl. Kollegen können es reproduzieren.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Dashboard zeigt falsche Summen", "Guten Tag,\n\ndie Summen im Dashboard sind heute falsch und passen nicht zur Aufgabenliste.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Upload bleibt bei 99% stehen", "Hallo,\n\nDatei-Uploads bleiben im Browser bei 99% hängen.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Mobile App stürzt ab", "Guten Tag,\n\ndie mobile App stürzt direkt nach dem Öffnen auf meinem Telefon ab.\n\n{name}", "bug", "normal", "mobile_app", False, None),
        ("Benachrichtigungen kommen nicht an", "Hallo,\n\nich erhalte seit gestern keine E-Mail-Benachrichtigungen mehr zu Aufgaben.\n\n{name}", "bug", "normal", "notifications", False, None),
        ("Slack-Integration synchronisiert nicht", "Guten Tag,\n\nunsere Slack-Integration sendet keine Updates mehr in den Kanal.\n\n{name}", "bug", "normal", "integrations", False, None),
        ("API-Webhook liefert 500", "Hallo,\n\nunser Webhook erhält 500-Fehler von der Nimbus-API bei Dokument-Ereignissen.\n\n{name}", "bug", "normal", "api", False, None),
        ("Gastberechtigungen werden nicht gespeichert", "Guten Tag,\n\nÄnderungen an Gastberechtigungen in der Admin-Konsole werden nicht gespeichert.\n\n{name}", "bug", "normal", "admin_console", False, None),
        ("Niemand kann auf Nimbus zugreifen", "Hallo,\n\nseit heute Morgen können sich bei uns alle nicht mehr anmelden. Niemand kann Nimbus öffnen. Gibt es einen Ausfall?\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Ausfall für alle Nutzer", "Guten Tag,\n\ngibt es gerade einen Ausfall? Unser ganzes Team kann Nimbus nicht öffnen.\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Ich sehe Rechnungen einer anderen Firma", "Hallo,\n\nich konnte die Rechnungen einer anderen Firma im Workspace-Wechsler sehen. Bitte dringend prüfen.\n\n{name}", "bug", "urgent", "admin_console", True, "data_exposure"),
        ("API-Schlüssel in öffentlichem Repo", "Guten Tag,\n\nwir haben versehentlich einen API-Schlüssel in einem öffentlichen GitHub-Repository veröffentlicht.\n\n{name}", "bug", "urgent", "api", True, "credential_leak"),

        ("Kalenderintegration gewünscht", "Hallo,\n\nkönnen Sie eine Integration mit Google Kalender hinzufügen?\n\n{name}", "feature_request", "low", "integrations", False, None),
        ("Benutzerdefinierte Aufgabenfelder", "Guten Tag,\n\nwir hätten gerne benutzerdefinierte Felder für Aufgaben.\n\n{name}", "feature_request", "low", "web_app", False, None),
        ("Dunkelmodus für mobile App", "Hallo,\n\nbitte fügen Sie einen Dunkelmodus für die mobile App hinzu.\n\n{name}", "feature_request", "low", "mobile_app", False, None),

        ("Wo kann ich die Rechnungs-E-Mail ändern?", "Kurze Frage: wo kann ich die E-Mail-Adresse für Rechnungen ändern? Ich finde die Option nicht.\n\n{name}", "other", "low", "billing", False, None),
        ("Wie aktualisiere ich die Zahlungskarte?", "Guten Tag,\n\nwie kann ich die Karte für monatliche Zahlungen aktualisieren?\n\n{name}", "other", "low", "billing", False, None),
        ("Wie lade ich alte Rechnungen herunter?", "Hallo,\n\nwo kann ich Rechnungen früherer Monate herunterladen?\n\n{name}", "other", "low", "billing", False, None),
        ("Allgemeine Frage zu Plänen", "Hallo,\n\nkönnen Sie die Unterschiede zwischen den Workspace-Plänen erklären?\n\n{name}", "other", "low", "general", False, None),
        ("Wie lade ich einen Gast ein?", "Guten Tag,\n\nwie kann ich einen Gast nur für einen Workspace einladen?\n\n{name}", "other", "low", "admin_console", False, None),
        ("Wie nutze ich Benachrichtigungseinstellungen?", "Hallo,\n\nwo kann ich auswählen, welche Aufgaben-Benachrichtigungen ich erhalte?\n\n{name}", "other", "low", "notifications", False, None),
    ])
    test = grouped([
        ("Neues Passwort wird nicht akzeptiert", "Guten Tag,\n\nnach dem Zurücksetzen wird mein neues Passwort weiterhin abgelehnt. Nur mein Konto ist betroffen.\n\n{name}", "login", "normal", "authentication", False, None),
        ("Kein Bestätigungscode beim Login", "Hallo,\n\nich erhalte beim Anmelden keinen MFA-Bestätigungscode. Kollegen sind nicht betroffen.\n\n{name}", "login", "normal", "authentication", False, None),

        ("Unerwartete Position auf Rechnung", "Guten Tag,\n\nauf der aktuellen Rechnung steht eine Position, die wir nicht zuordnen können. Bitte prüfen.\n\n{name}", "billing", "normal", "billing", False, None),
        ("Zweimal abgerechnet", "Hallo,\n\nunser Workspace wurde in diesem Abrechnungszeitraum offenbar doppelt belastet.\n\n{name}", "billing", "normal", "billing", False, None),

        ("Berichtsexport liefert Fehler", "Hallo,\n\nbeim Export eines Berichts im Web-App tritt bei mehreren Nutzern ein Fehler auf.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Anhang-Upload endet nie", "Guten Tag,\n\nder Upload von Anhängen im Browser wird nicht abgeschlossen.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Nimbus-Ausfall im gesamten Büro", "Hallo,\n\nes gibt einen Ausfall: niemand im Büro kann sich bei Nimbus anmelden. Die Plattform ist derzeit nicht verfügbar.\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Rechnungen einer anderen Firma sichtbar", "Guten Tag,\n\nich kann die Rechnungen einer anderen Firma im Workspace sehen. Bitte dringend prüfen.\n\n{name}", "bug", "urgent", "admin_console", True, "data_exposure"),
        ("Zugriffstoken in öffentlichem Repository", "Hallo,\n\nein Zugriffstoken wurde versehentlich in einem öffentlichen GitHub-Repository veröffentlicht.\n\n{name}", "bug", "urgent", "api", True, "credential_leak"),

        ("Wunsch: Aufgabenvorlagen", "Hallo,\n\nwäre es möglich, wiederverwendbare Aufgabenvorlagen bereitzustellen?\n\n{name}", "feature_request", "low", "web_app", False, None),
        ("Vorschlag: Audit-Log exportieren", "Guten Tag,\n\nes wäre hilfreich, Audit-Logs aus der Admin-Konsole exportieren zu können.\n\n{name}", "feature_request", "low", "admin_console", False, None),

        ("Rechnungsempfänger ändern", "Hallo,\n\nwo kann ich ändern, welche E-Mail-Adresse unsere Rechnungen erhält?\n\n{name}", "other", "low", "billing", False, None),
        ("Zahlungsmethode bearbeiten", "Guten Tag,\n\nkönnen Sie mir sagen, wo ich die Zahlungsmethode für unser Abo ändere?\n\n{name}", "other", "low", "billing", False, None),
    ])
    return train, test


def cases_fr():
    train = grouped([
        ("Connexion impossible après réinitialisation", "Bonjour,\n\nj'ai réinitialisé mon mot de passe mais je ne peux toujours pas me connecter. Seul mon compte est concerné.\n\n{name}", "login", "normal", "authentication", False, None),
        ("Code MFA non reçu", "Bonjour,\n\nje n'arrive pas à recevoir le code MFA sur mon téléphone. Personne d'autre dans l'équipe n'a ce problème.\n\n{name}", "login", "normal", "authentication", False, None),
        ("Échec SSO pour mon compte", "Bonjour,\n\nla connexion SSO me renvoie vers la page de connexion uniquement pour mon compte.\n\n{name}", "login", "normal", "authentication", False, None),

        ("Montant de facture incorrect", "Bonjour,\n\nnotre dernière facture contient un montant inattendu. Pouvez-vous vérifier ?\n\n{name}", "billing", "normal", "billing", False, None),
        ("Double débit sur facture", "Bonjour,\n\nnous avons été débités deux fois ce mois-ci. Merci de corriger la facture.\n\n{name}", "billing", "normal", "billing", False, None),
        ("Facture du mois dernier manquante", "Bonjour,\n\nje ne trouve pas la facture du mois dernier dans la section facturation.\n\n{name}", "billing", "normal", "billing", False, None),

        ("L'export échoue dans le navigateur", "Bonjour,\n\nl'export d'un document échoue dans l'application web. Mes collègues peuvent reproduire le problème.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Le tableau de bord affiche de mauvais totaux", "Bonjour,\n\nles totaux du tableau de bord sont incorrects aujourd'hui.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Téléversement bloqué à 99%", "Bonjour,\n\nles téléversements de fichiers restent bloqués à 99% dans le navigateur.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("L'application mobile plante", "Bonjour,\n\nl'application mobile Nimbus se ferme immédiatement à l'ouverture sur mon téléphone.\n\n{name}", "bug", "normal", "mobile_app", False, None),
        ("Notifications non reçues", "Bonjour,\n\nje ne reçois plus les notifications par e-mail pour les tâches depuis hier.\n\n{name}", "bug", "normal", "notifications", False, None),
        ("L'intégration Slack ne synchronise plus", "Bonjour,\n\nnotre intégration Slack ne publie plus les mises à jour dans le canal.\n\n{name}", "bug", "normal", "integrations", False, None),
        ("Webhook API renvoie 500", "Bonjour,\n\nnotre webhook reçoit des erreurs 500 de l'API Nimbus pour les événements de documents.\n\n{name}", "bug", "normal", "api", False, None),
        ("Permissions des invités non enregistrées", "Bonjour,\n\nles modifications des permissions des invités dans la console d'administration ne sont pas enregistrées.\n\n{name}", "bug", "normal", "admin_console", False, None),
        ("Personne ne peut accéder à Nimbus", "Bonjour,\n\npersonne ne peut se connecter à Nimbus depuis ce matin dans toute notre équipe. Est-ce une panne en cours ?\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Panne pour toute l'équipe", "Bonjour,\n\ntoute l'équipe ne peut plus accéder à Nimbus en ce moment. Personne ne peut se connecter.\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Je vois les données d'une autre entreprise", "Bonjour,\n\nje vois les données d'une autre entreprise dans le sélecteur d'espace de travail.\n\n{name}", "bug", "urgent", "admin_console", True, "data_exposure"),
        ("Clé API dans un dépôt public", "Bonjour,\n\nnous avons commis une clé API dans un dépôt GitHub public par erreur.\n\n{name}", "bug", "urgent", "api", True, "credential_leak"),

        ("Ajouter une intégration calendrier", "Bonjour,\n\npouvez-vous ajouter une synchronisation avec Google Agenda ?\n\n{name}", "feature_request", "low", "integrations", False, None),
        ("Champs personnalisés pour les tâches", "Bonjour,\n\nserait-il possible d'ajouter des champs personnalisés aux tâches ?\n\n{name}", "feature_request", "low", "web_app", False, None),
        ("Mode sombre pour l'application mobile", "Bonjour,\n\nmerci d'envisager un mode sombre pour l'application mobile.\n\n{name}", "feature_request", "low", "mobile_app", False, None),

        ("Où changer l'e-mail de facturation ?", "Question rapide : où puis-je changer l'adresse e-mail qui reçoit les factures ?\n\n{name}", "other", "low", "billing", False, None),
        ("Comment mettre à jour la carte de paiement ?", "Bonjour,\n\ncomment mettre à jour la carte utilisée pour les paiements mensuels ?\n\n{name}", "other", "low", "billing", False, None),
        ("Comment télécharger d'anciennes factures ?", "Bonjour,\n\noù puis-je télécharger les factures des mois précédents ?\n\n{name}", "other", "low", "billing", False, None),
        ("Question générale sur les forfaits", "Bonjour,\n\npouvez-vous expliquer la différence entre vos forfaits d'espace de travail ?\n\n{name}", "other", "low", "general", False, None),
        ("Comment inviter un invité ?", "Bonjour,\n\ncomment inviter un invité dans un seul espace de travail ?\n\n{name}", "other", "low", "admin_console", False, None),
        ("Comment utiliser les paramètres de notification ?", "Bonjour,\n\noù puis-je choisir les notifications de tâches que je reçois ?\n\n{name}", "other", "low", "notifications", False, None),
    ])
    test = grouped([
        ("Nouveau mot de passe refusé", "Bonjour,\n\naprès réinitialisation, mon nouveau mot de passe est toujours refusé. Seul mon compte est impacté.\n\n{name}", "login", "normal", "authentication", False, None),
        ("Code de vérification non reçu", "Bonjour,\n\nje ne reçois jamais le code de vérification MFA à la connexion. Les autres ne sont pas concernés.\n\n{name}", "login", "normal", "authentication", False, None),

        ("Ligne inattendue sur la facture", "Bonjour,\n\nla facture actuelle comporte une ligne que nous ne reconnaissons pas. Pouvez-vous vérifier ?\n\n{name}", "billing", "normal", "billing", False, None),
        ("Facturation en double ce cycle", "Bonjour,\n\nnotre espace de travail semble avoir été facturé deux fois sur ce cycle.\n\n{name}", "billing", "normal", "billing", False, None),

        ("Erreur lors de l'export du rapport", "Bonjour,\n\nl'export d'un rapport dans l'application web échoue pour plusieurs utilisateurs.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("L'ajout de pièce jointe ne se termine pas", "Bonjour,\n\nle téléversement de pièces jointes dans le navigateur ne se termine jamais.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Nimbus inaccessible pour tout le bureau", "Bonjour,\n\npersonne ne peut se connecter à Nimbus au bureau. Le service est indisponible pour tous.\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Factures d'un autre client visibles", "Bonjour,\n\nje vois les données d'une autre entreprise dans l'espace de travail. Merci de vérifier en urgence.\n\n{name}", "bug", "urgent", "admin_console", True, "data_exposure"),
        ("Jeton d'accès exposé dans un dépôt public", "Bonjour,\n\nun jeton d'accès a été publié dans un dépôt GitHub public par erreur.\n\n{name}", "bug", "urgent", "api", True, "credential_leak"),

        ("Demande : modèles de tâches", "Bonjour,\n\nserait-il possible d'ajouter des modèles de tâches réutilisables ?\n\n{name}", "feature_request", "low", "web_app", False, None),
        ("Suggestion : export du journal d'audit", "Bonjour,\n\nil serait utile de pouvoir exporter le journal d'audit depuis la console d'administration.\n\n{name}", "feature_request", "low", "admin_console", False, None),

        ("Modifier le destinataire des factures", "Bonjour,\n\noù puis-je modifier l'adresse e-mail qui reçoit nos factures ?\n\n{name}", "other", "low", "billing", False, None),
        ("Où changer le moyen de paiement ?", "Bonjour,\n\npouvez-vous m'indiquer où modifier le moyen de paiement de notre abonnement ?\n\n{name}", "other", "low", "billing", False, None),
    ])
    return train, test


def cases_es():
    train = grouped([
        ("No puedo iniciar sesión tras restablecer", "Hola,\n\nrestablecí mi contraseña pero Nimbus sigue diciendo que es incorrecta. Solo me afecta a mí.\n\n{name}", "login", "normal", "authentication", False, None),
        ("No llega el código MFA", "Hola,\n\nestoy intentando iniciar sesión y el código MFA no llega a mi teléfono. Nadie más del equipo lo reportó.\n\n{name}", "login", "normal", "authentication", False, None),
        ("Fallo de SSO en mi cuenta", "Hola,\n\nSSO me devuelve a la página de inicio solo en mi cuenta. ¿Podéis revisarlo?\n\n{name}", "login", "normal", "authentication", False, None),

        ("Importe de factura incorrecto", "Hola,\n\nnuestra última factura incluye un cargo que no reconocemos. ¿Podéis revisarlo?\n\n{name}", "billing", "normal", "billing", False, None),
        ("Cargo duplicado en factura", "Hola,\n\nnos han cobrado dos veces en la factura de este mes. Por favor, corregidlo.\n\n{name}", "billing", "normal", "billing", False, None),
        ("Falta la factura del mes pasado", "Hola,\n\nno encuentro la factura del mes pasado en la sección de facturación.\n\n{name}", "billing", "normal", "billing", False, None),

        ("La exportación falla en el navegador", "Hola,\n\nal exportar un documento desde la aplicación web falla con error. Mis compañeros pueden reproducirlo.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("El panel muestra totales incorrectos", "Hola,\n\nlos totales del panel son incorrectos hoy y no coinciden con las tareas.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("La subida se queda en 99%", "Hola,\n\nlas subidas de archivos se quedan atascadas al 99% en el navegador.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("La app móvil se cierra", "Hola,\n\nla aplicación móvil de Nimbus se cierra al abrirla en mi teléfono.\n\n{name}", "bug", "normal", "mobile_app", False, None),
        ("No llegan notificaciones", "Hola,\n\ndesde ayer no recibo notificaciones por correo de las tareas.\n\n{name}", "bug", "normal", "notifications", False, None),
        ("La integración con Slack no sincroniza", "Hola,\n\nnuestra integración con Slack ya no publica actualizaciones en el canal.\n\n{name}", "bug", "normal", "integrations", False, None),
        ("Webhook de la API devuelve 500", "Hola,\n\nnuestro webhook recibe errores 500 de la API de Nimbus en eventos de documentos.\n\n{name}", "bug", "normal", "api", False, None),
        ("Permisos de invitados no se guardan", "Hola,\n\nlos cambios de permisos de invitados en la consola de administración no se guardan.\n\n{name}", "bug", "normal", "admin_console", False, None),
        ("Nadie puede acceder a Nimbus", "Hola,\n\nnadie puede iniciar sesión en Nimbus desde esta mañana en toda nuestra oficina. ¿Hay una caída del servicio en curso?\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Caída del servicio para todos", "Hola,\n\ntodo el equipo está sin poder acceder a Nimbus ahora mismo. Nadie puede iniciar sesión.\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Veo los datos de otra empresa", "Hola,\n\nveo los datos de otra empresa en el selector de espacios. Por favor, revisadlo con urgencia.\n\n{name}", "bug", "urgent", "admin_console", True, "data_exposure"),
        ("Clave API en repositorio público", "Hola,\n\nhemos subido por error una clave API a un repositorio público de GitHub.\n\n{name}", "bug", "urgent", "api", True, "credential_leak"),

        ("Agregar integración con calendario", "Hola,\n\n¿podríais añadir sincronización con Google Calendar?\n\n{name}", "feature_request", "low", "integrations", False, None),
        ("Campos personalizados en tareas", "Hola,\n\nnos gustaría tener campos personalizados en las tareas.\n\n{name}", "feature_request", "low", "web_app", False, None),
        ("Modo oscuro para la app móvil", "Hola,\n\npor favor, considerad añadir modo oscuro a la aplicación móvil.\n\n{name}", "feature_request", "low", "mobile_app", False, None),

        ("¿Dónde cambio el correo de facturación?", "Pregunta rápida: ¿dónde puedo cambiar el correo que recibe las facturas? No lo encuentro.\n\n{name}", "other", "low", "billing", False, None),
        ("¿Cómo actualizo la tarjeta de pago?", "Hola,\n\n¿cómo actualizo la tarjeta usada para los pagos mensuales?\n\n{name}", "other", "low", "billing", False, None),
        ("¿Cómo descargo facturas antiguas?", "Hola,\n\n¿dónde puedo descargar facturas de meses anteriores?\n\n{name}", "other", "low", "billing", False, None),
        ("Pregunta general sobre planes", "Hola,\n\n¿podéis explicar la diferencia entre los planes de espacio de trabajo?\n\n{name}", "other", "low", "general", False, None),
        ("¿Cómo invito a un invitado?", "Hola,\n\n¿cómo puedo invitar a un invitado solo a un espacio de trabajo?\n\n{name}", "other", "low", "admin_console", False, None),
        ("¿Cómo uso los ajustes de notificaciones?", "Hola,\n\n¿dónde puedo elegir qué notificaciones de tareas recibo?\n\n{name}", "other", "low", "notifications", False, None),
    ])
    test = grouped([
        ("La nueva contraseña no funciona", "Hola,\n\ntras restablecerla, mi nueva contraseña sigue sin funcionar para iniciar sesión. Solo es mi cuenta.\n\n{name}", "login", "normal", "authentication", False, None),
        ("No recibo el código de verificación", "Hola,\n\nno recibo el código de verificación MFA al intentar entrar. Nadie más tiene este problema.\n\n{name}", "login", "normal", "authentication", False, None),

        ("Cargo inesperado en la factura", "Hola,\n\nla factura actual incluye un cargo que no reconocemos. ¿Podéis revisarlo?\n\n{name}", "billing", "normal", "billing", False, None),
        ("Cobro doble en este ciclo", "Hola,\n\nparece que nos han cobrado dos veces en este ciclo de facturación.\n\n{name}", "billing", "normal", "billing", False, None),

        ("Error al exportar informe", "Hola,\n\nal exportar un informe en la aplicación web aparece un error para varios usuarios.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("La carga de adjuntos no termina", "Hola,\n\nla subida de archivos adjuntos en el navegador nunca se completa.\n\n{name}", "bug", "normal", "web_app", False, None),
        ("Nimbus no disponible para toda la oficina", "Hola,\n\nnadie puede iniciar sesión en Nimbus ahora mismo en toda la oficina. Hay una caída del servicio en curso.\n\n{name}", "bug", "urgent", "web_app", True, "service_wide_outage"),
        ("Veo facturas de otra empresa", "Hola,\n\nveo los datos de otra empresa en el selector de espacios. Revisadlo urgente.\n\n{name}", "bug", "urgent", "admin_console", True, "data_exposure"),
        ("Token de acceso expuesto en repositorio público", "Hola,\n\nun token de acceso fue publicado por error en un repositorio público de GitHub.\n\n{name}", "bug", "urgent", "api", True, "credential_leak"),

        ("Solicitud: plantillas de tareas", "Hola,\n\n¿sería posible añadir plantillas de tareas reutilizables?\n\n{name}", "feature_request", "low", "web_app", False, None),
        ("Sugerencia: exportar registro de auditoría", "Hola,\n\nsería útil poder exportar el registro de auditoría desde la consola de administración.\n\n{name}", "feature_request", "low", "admin_console", False, None),

        ("Cambiar destinatario de facturas", "Hola,\n\n¿dónde puedo cambiar qué correo recibe nuestras facturas?\n\n{name}", "other", "low", "billing", False, None),
        ("Dónde editar el método de pago", "Hola,\n\n¿podéis indicarme dónde editar el método de pago de la suscripción?\n\n{name}", "other", "low", "billing", False, None),
    ])
    return train, test


NAMES = {
    "en": ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Chris"],
    "de": ["Lena", "Jonas", "Mia", "Paul", "Sofia", "Leon", "Emma", "Noah", "Laura", "Ben"],
    "fr": ["Camille", "Lucas", "Emma", "Louis", "Chloe", "Hugo", "Lea", "Noah", "Jade", "Leo"],
    "es": ["Lucia", "Mateo", "Sofia", "Diego", "Valeria", "Pablo", "Carmen", "Hugo", "Elena", "Adrian"],
}


def pick_bug_case(by_type, index, escalation_pattern):
    normal = [c for c in by_type["bug"] if not c[5]]
    escalation_by_rule = defaultdict(list)
    for c in by_type["bug"]:
        if c[5]:
            escalation_by_rule[c[6]].append(c)
    rule = escalation_pattern[index] if index < len(escalation_pattern) else None
    if rule is None:
        return normal[index % len(normal)]
    options = escalation_by_rule[rule]
    return options[index % len(options)]


def build_split(by_type, count_per_type, lang, split_name, rng, escalation_pattern, name_start):
    tickets, labels = [], []
    seq = 1
    for issue_type in ISSUE_TYPES:
        templates = by_type[issue_type]
        for i in range(count_per_type):
            if issue_type == "bug":
                case = pick_bug_case(by_type, i, escalation_pattern)
            else:
                case = templates[i % len(templates)]
            subject, body_tpl, issue_t, priority, product_area, escalation_required, escalation_rule = case
            assert issue_t == issue_type
            name = NAMES[lang][(name_start + seq) % len(NAMES[lang])]
            ticket_id = f"{lang}-{split_name}-{seq:03d}"
            ticket = {
                "id": ticket_id,
                "from": f"{name.lower()}.{lang}.{split_name[0]}{seq}@example.com",
                "created_at": f"2026-11-20T{8 + (seq % 10):02d}:{(seq * 7) % 60:02d}:00Z",
                "subject": subject,
                "body": body_tpl.format(name=name),
            }
            label = {
                "id": ticket_id,
                "language": lang,
                "issue_type": issue_type,
                "priority": priority,
                "product_area": product_area,
                "escalation_required": escalation_required,
                "escalation_rule": escalation_rule,
            }
            tickets.append(ticket)
            labels.append(label)
            seq += 1
    combined = list(zip(tickets, labels))
    rng.shuffle(combined)
    tickets, labels = zip(*combined)
    return list(tickets), list(labels)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    all_langs = [("en", cases_en()), ("de", cases_de()), ("fr", cases_fr()), ("es", cases_es())]
    train_tickets, train_labels, test_tickets, test_labels = [], [], [], []
    for lang, (train_by_type, test_by_type) in all_langs:
        for split_name, by_type in [("train", train_by_type), ("test", test_by_type)]:
            missing = [t for t in ISSUE_TYPES if t not in by_type]
            if missing:
                raise SystemExit(f"{lang} {split_name} missing templates for {missing}")
            # ensure escalation rules available
            if split_name == "train":
                needed = set(TRAIN_ESCALATION_PATTERN)
            else:
                needed = {r for r in TEST_ESCALATION_PATTERN if r}
            have = {c[6] for c in by_type["bug"] if c[5]}
            if not needed.issubset(have):
                raise SystemExit(f"{lang} {split_name} missing escalation templates {needed-have}")
        rng_train = random.Random(f"{RANDOM_SEED}-{lang}-train")
        rng_test = random.Random(f"{RANDOM_SEED}-{lang}-test")
        tr_t, tr_l = build_split(train_by_type, TRAIN_PER_TYPE, lang, "train", rng_train, TRAIN_ESCALATION_PATTERN, name_start=0)
        te_t, te_l = build_split(test_by_type, TEST_PER_TYPE, lang, "test", rng_test, TEST_ESCALATION_PATTERN, name_start=3)
        train_tickets.extend(tr_t)
        train_labels.extend(tr_l)
        test_tickets.extend(te_t)
        test_labels.extend(te_l)

    rng = random.Random(RANDOM_SEED)
    for pair in [(train_tickets, train_labels), (test_tickets, test_labels)]:
        combined = list(zip(pair[0], pair[1]))
        rng.shuffle(combined)
        pair[0][:], pair[1][:] = zip(*combined)

    DATA.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATA / "train_tickets.jsonl", train_tickets)
    write_jsonl(DATA / "train_labels.jsonl", train_labels)
    write_jsonl(DATA / "test_tickets.jsonl", test_tickets)
    write_jsonl(DATA / "test_labels.jsonl", test_labels)

    print("train", len(train_tickets), "test", len(test_tickets))
    for name, labels in [("train", train_labels), ("test", test_labels)]:
        print(name, "issue_type", dict(sorted(Counter(l["issue_type"] for l in labels).items())))
        print(name, "language", dict(sorted(Counter(l["language"] for l in labels).items())))
        esc = Counter(l["escalation_rule"] for l in labels)
        print(
            name,
            "escalation",
            {
                ("None" if k is None else k): v
                for k, v in sorted(esc.items(), key=lambda x: (x[0] is not None, str(x[0])))
            },
        )


if __name__ == "__main__":
    main()
