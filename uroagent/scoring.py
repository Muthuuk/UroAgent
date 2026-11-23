"""ScoringAgent: symptom severity scores.

Scoring structure follows the published instruments (ICIQ-OAB: four items scored 0-4,
total 0-16; O'Leary-Sant: symptom index 0-20 and problem index 0-16). Question wording
is not reproduced here: the ICIQ and O'Leary-Sant questionnaires are licensed and must
be obtained from their owners. Scores estimated from free text are approximate and are
labelled as estimates.
"""


def _band(score, cuts):
    for limit, name in cuts:
        if score <= limit:
            return name
    return cuts[-1][1]


def iciq_oab(frequency, nocturia, urgency, urge_leak):
    """Each item 0-4. Returns total 0-16 with a descriptive band."""
    items = [int(min(max(v, 0), 4)) for v in (frequency, nocturia, urgency, urge_leak)]
    total = sum(items)
    return {"instrument": "ICIQ-OAB (structure only)", "items": dict(zip(["frequency", "nocturia", "urgency", "urge_leak"], items)),
            "score": total, "max": 16, "band": _band(total, [(4, "mild"), (9, "moderate"), (16, "severe")])}


def oleary_sant(icsi_items, icpi_items):
    """icsi_items: urgency, frequency, nocturia, pain (each 0-5); icpi_items: four items 0-4."""
    icsi = sum(int(min(max(v, 0), 5)) for v in icsi_items)
    icpi = sum(int(min(max(v, 0), 4)) for v in icpi_items)
    return {"instrument": "O'Leary-Sant ICSI/ICPI (structure only)", "icsi": icsi, "icsi_max": 20,
            "icpi": icpi, "icpi_max": 16, "total": icsi + icpi,
            "band": _band(icsi + icpi, [(11, "mild"), (23, "moderate"), (36, "severe")])}


def estimate_from_entities(e: dict) -> dict:
    """Approximate item levels from extracted entities (clearly labelled as estimates)."""
    f = e["day_frequency"]
    freq_item = 0 if f <= 6 else 1 if f <= 8 else 2 if f <= 10 else 3 if f <= 12 else 4
    noct_item = int(min(e["nocturia"], 4))
    urg_item = 3 if e["urgency"] else 0
    leak_item = 2 if e["urge_incontinence"] else 0
    i = iciq_oab(freq_item, noct_item, urg_item, leak_item)
    pain = e["pain_score"]
    pain5 = int(round(min(pain, 10) / 2))
    o = oleary_sant([4 if e["urgency"] else 1, min(freq_item + 1, 5), min(noct_item + 1, 5), pain5],
                    [min(freq_item, 4), min(noct_item, 4), 3 if e["urgency"] else 0, min(int(round(pain / 2.5)), 4)])
    i["estimated"] = o["estimated"] = True
    return {"iciq_oab": i, "oleary_sant": o}
