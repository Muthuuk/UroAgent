"""Hard safety rules that sit outside the models and always win (case study 7.3, 7.5, 9.1).

These are illustrative rules for a research demonstrator, written so a clinician can
read every one. They must be reviewed and signed off by a clinical safety officer
before any real use.
"""
import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions", r"disregard (the )?(system|previous)",
    r"you are now", r"act as (an? )?(unrestricted|different)", r"reveal (your|the) (system )?prompt",
    r"jailbreak", r"developer mode",
]

OUT_OF_SCOPE = {
    "mental health": ["suicid", "self-harm", "self harm", "depress", "anxiety", "psychosis"],
    "oncology": ["chemotherapy", "tumour staging", "tumor staging", "metasta", "radiotherapy plan"],
    "cardiac emergency": ["chest pain", "heart attack", "cardiac arrest", "stroke"],
    "a specialty or group outside this tool (for example pregnancy, children or non-urological conditions)": ["pregnan", "child under", "paediatric", "pediatric", "fracture", "asthma", "diabetes management"],
}

RED_FLAG_RULES = [
    ("visible haematuria", [r"visible (haematuria|hematuria|blood)", r"frank (haematuria|hematuria)", r"blood clots? in (the )?urine", r"red urine"],
     "Visible blood in the urine: consider the suspected cancer pathway (for example NICE NG12 for adults aged 45 and over). Do not attribute to OAB, BPS or simple UTI without assessment."),
    ("possible pyelonephritis or sepsis", [r"loin pain", r"flank pain", r"rigors?", r"shaking", r"(temperature|temp|fever)[^.]{0,20}(38|39|40)", r"vomiting", r"confus"],
     "Signs of upper urinary tract infection or sepsis: assess urgently and follow local sepsis and pyelonephritis pathways."),
    ("urinary retention", [r"unable to (pass|void|wee)", r"can'?t pass urine", r"retention", r"painful distended bladder"],
     "Possible urinary retention: needs same-day assessment."),
    ("possible cauda equina", [r"saddle (numbness|anaesthesia|anesthesia)", r"(new|sudden) (leg weakness|numbness)", r"back pain[^.]{0,40}(numb|weak)"],
     "Neurological red flags with bladder change: emergency assessment for cauda equina syndrome."),
    ("UTI in a man or recurrent upper tract", [r"\b(man|male|he|his)\b[^.]{0,80}(burning|dysuria|uti)"],
     "Urinary infection in a man is treated as complicated: send urine culture and follow local guidance."),
]


def detect_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def out_of_scope_reason(text: str):
    t = text.lower()
    for area, words in OUT_OF_SCOPE.items():
        if any(w in t for w in words):
            return area
    return None


def red_flags(text: str):
    """Return a list of (name, advice) for every red-flag rule that fires."""
    t = text.lower()
    hits = []
    for name, patterns, advice in RED_FLAG_RULES:
        if any(re.search(p, t) for p in patterns):
            hits.append((name, advice))
    return hits
