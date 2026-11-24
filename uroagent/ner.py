"""Rule-based clinical entity extraction (the 'NER' step of the TriageAgent).

Reads a free-text urology note and returns structured features. Rules are simple and
transparent so they can be audited; an optional language model can be added later
without changing the output format.
"""
import re

FEATURES = ["age", "female", "duration_weeks", "day_frequency", "nocturia", "urgency",
            "urge_incontinence", "dysuria", "suprapubic_pain", "pain_relief_on_voiding",
            "pain_on_filling", "pain_score", "fever", "cloudy_urine", "nitrite", "leukocytes",
            "acute_onset", "dietary_triggers", "negative_cultures", "recurrent_uti"]

LABELS = {
    "age": "Age", "female": "Female sex", "duration_weeks": "Symptom duration (weeks)",
    "day_frequency": "Daytime frequency (per day)", "nocturia": "Nocturia (per night)",
    "urgency": "Urgency", "urge_incontinence": "Urge incontinence", "dysuria": "Burning on passing urine",
    "suprapubic_pain": "Bladder or pelvic pain", "pain_relief_on_voiding": "Pain eased by emptying bladder",
    "pain_on_filling": "Pain worse as bladder fills", "pain_score": "Pain score (0-10)",
    "fever": "Fever", "cloudy_urine": "Cloudy or smelly urine", "nitrite": "Dipstick nitrite positive",
    "leukocytes": "Dipstick leukocytes positive", "acute_onset": "Sudden onset",
    "dietary_triggers": "Flares with coffee, citrus or spicy food", "negative_cultures": "Previous negative urine cultures",
    "recurrent_uti": "History of recurrent UTI",
}


def _num(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def _first(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default


def _any(text, words):
    return int(any(w in text for w in words))


NEGATION = re.compile(r"\b(no|not|denies|denied|without|never|nil|negative for|free of|absence of)\b[^,.;]*")
NEGATION_KEEP = re.compile(r"\b(?:not|no longer|never) able to\b|\bcan ?not\b|\bcan'?t\b")


def remove_negated(text: str) -> str:
    """Drop clauses that are negated ('no fever', 'denies burning') so they are not read as
    present. Phrases that mean a problem ('not able to pass urine') are kept."""
    keep = NEGATION_KEEP.sub("unable to", text).replace("unable to to", "unable to")
    return NEGATION.sub(" ", keep)


def extract_entities(note: str) -> dict:
    raw = " " + note.lower().replace("\n", " ") + " "
    t = remove_negated(raw)
    e = {}
    e["age"] = _first(_num(r"(\d{2})[- ]?(?:year|yr)s?[- ]old", t, int), _num(r"\baged? (\d{2})", t, int), default=45)
    e["female"] = 0 if re.search(r"\b(man|male|gentleman|he|his)\b", t) else 1
    weeks = _num(r"(\d+)\s*weeks?", t)
    months = _num(r"(\d+)\s*months?", t)
    days = _num(r"(\d+)\s*days?", t)
    years = _num(r"(\d+)\s*years?(?! old)", t)
    e["duration_weeks"] = (weeks if weeks is not None else
                           months * 4.3 if months is not None else
                           years * 52 if years is not None else
                           days / 7 if days is not None else 8)
    e["day_frequency"] = _first(_num(r"(\d+)\s*times\s*(?:a|per|each)\s*day", t),
                               _num(r"every (\d+) hours?", t, lambda h: round(16 / max(float(h), 0.5))),
                               default=(12 if "frequency" in t or "frequent" in t else 6))
    e["nocturia"] = _first(_num(r"(\d+)\s*times\s*(?:at|a|per|each)\s*night", t), _num(r"up (\d+) times", t),
                          default=(2 if "nocturia" in t or "at night" in t else 0))
    e["urgency"] = _any(t, ["urgen", "sudden need", "can't wait", "cannot wait", "rush to", "can't hold", "hold it"])
    e["urge_incontinence"] = _any(t, ["leak", "wet herself", "wets", "incontinen", "accidents"])
    e["dysuria"] = _any(t, ["burning", "burns", "dysuria", "sting", "painful to pass", "hurts to pass", "hurts when i wee", "hurts when i pee", "pain when i wee", "pain when i pee"])
    e["suprapubic_pain"] = _any(t, ["bladder pain", "pelvic pain", "suprapubic", "lower tummy pain", "lower abdominal pain", "pressure"])
    e["pain_relief_on_voiding"] = _any(t, ["eased by passing", "relieved by voiding", "better after emptying", "eases after", "relief after passing"])
    e["pain_on_filling"] = _any(t, ["as the bladder fills", "worse when full", "builds up as", "worse as bladder fills"])
    e["pain_score"] = _first(_num(r"pain\D{0,12}(\d{1,2})\s*/\s*10", t), _num(r"(\d{1,2})\s*/\s*10", t),
                            default=(5 if e["suprapubic_pain"] else 0))
    temp = _num(r"(3[6-9]\.\d|4[0-1]\.\d)", raw)
    e["fever"] = int((temp is not None and temp >= 37.8) or "fever" in t or "feverish" in t)
    e["cloudy_urine"] = _any(t, ["cloudy", "smelly", "offensive urine", "foul", "strong smell"])
    e["nitrite"] = int(bool(re.search(r"nitrites?\s*(positive|\+|present)", raw)) or "nitrite positive" in raw)
    e["leukocytes"] = int(bool(re.search(r"(leukocytes?|leucocytes?)\s*(positive|\+|present|\d\+)", raw)) or "leukocyte positive" in raw)
    e["acute_onset"] = _any(t, ["sudden onset", "started suddenly", "came on suddenly", "since yesterday", "two days", "2 days", "3 days", "acute"])
    e["dietary_triggers"] = _any(t, ["coffee", "citrus", "spicy", "tomato", "alcohol makes", "acidic"])
    e["negative_cultures"] = _any(t, ["negative culture", "cultures negative", "cultures were negative", "no growth", "repeatedly negative"])
    e["recurrent_uti"] = _any(t, ["recurrent uti", "recurrent infections", "several utis", "frequent utis"])
    return {k: float(e[k]) for k in FEATURES}


def describe(entities: dict) -> list:
    """Human-readable list of the findings that are present."""
    out = []
    for k, v in entities.items():
        if k in ("age", "female", "duration_weeks", "day_frequency", "nocturia", "pain_score"):
            continue
        if v:
            out.append(LABELS[k])
    return out
