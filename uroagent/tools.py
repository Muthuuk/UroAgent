"""Clinical tools exposed in Model Context Protocol (MCP) style (case study 5.4).

Each tool is a plain, documented Python function with JSON-friendly inputs and outputs,
listed in TOOLS so the agent (or an MCP server, see mcp_server.py) can call it by name.
All patient data here are synthetic.
"""
import hashlib
import json
import re
import time
from pathlib import Path

from .rag import KB_PATH
from . import scoring

AUDIT_PATH = Path("uroagent_audit.jsonl")

_FHIR = {
    "demo-001": {"age": 34, "sex": "female", "conditions": ["recurrent UTI"], "meds": ["none"], "egfr": 95, "bp": "118/74"},
    "demo-002": {"age": 71, "sex": "female", "conditions": ["overactive bladder", "hypertension"], "meds": ["oxybutynin", "amlodipine", "amitriptyline"], "egfr": 58, "bp": "152/88"},
    "demo-003": {"age": 46, "sex": "female", "conditions": ["bladder pain syndrome"], "meds": ["amitriptyline"], "egfr": 88, "bp": "124/80"},
}


def fhir_patient_query(patient_id: str) -> dict:
    """Return a small synthetic FHIR R4-style Bundle for a pseudonymised patient id."""
    p = _FHIR.get(patient_id)
    if not p:
        return {"resourceType": "Bundle", "total": 0, "entry": [], "note": "unknown synthetic patient"}
    entries = [{"resource": {"resourceType": "Patient", "id": patient_id, "gender": p["sex"], "extension": [{"url": "age", "valueInteger": p["age"]}]}}]
    entries += [{"resource": {"resourceType": "Condition", "code": {"text": c}}} for c in p["conditions"]]
    entries += [{"resource": {"resourceType": "MedicationStatement", "medicationCodeableConcept": {"text": m}}} for m in p["meds"]]
    entries += [{"resource": {"resourceType": "Observation", "code": {"text": "eGFR"}, "valueQuantity": {"value": p["egfr"], "unit": "mL/min/1.73m2"}}},
                {"resource": {"resourceType": "Observation", "code": {"text": "Blood pressure"}, "valueString": p["bp"]}}]
    return {"resourceType": "Bundle", "type": "searchset", "total": len(entries), "entry": entries}


def iciq_score_calculator(frequency: int, nocturia: int, urgency: int, urge_leak: int) -> dict:
    """ICIQ-OAB structure: four items scored 0-4 (total 0-16)."""
    return scoring.iciq_oab(frequency, nocturia, urgency, urge_leak)


def bpss_score_calculator(icsi_items: list, icpi_items: list) -> dict:
    """Bladder pain severity using the O'Leary-Sant structure (ICSI 0-20, ICPI 0-16)."""
    return scoring.oleary_sant(icsi_items, icpi_items)


def lab_result_parser(report: str) -> dict:
    """Parse a simple urinalysis / culture / creatinine text report."""
    t = report.lower()
    def pos(name):
        m = re.search(name + r"\w*\s*[:=]?\s*(negative|trace|positive|\+{1,3}|[1-3]\+)", t)
        return None if not m else ("negative" if m.group(1) == "negative" else "positive")
    culture = re.search(r"culture\s*[:=]?\s*([^;\n]+)", t)
    creat = re.search(r"creatinine\s*[:=]?\s*(\d+)", t)
    return {"nitrite": pos("nitrite"), "leukocytes": pos("leu[ck]ocyte"), "blood": pos("blood"),
            "culture": culture.group(1).strip() if culture else None,
            "creatinine_umol_l": int(creat.group(1)) if creat else None}


ANTICHOLINERGICS = {"oxybutynin", "tolterodine", "solifenacin", "fesoterodine", "trospium", "darifenacin", "propiverine"}
OTHER_ACB = {"amitriptyline", "chlorphenamine", "hydroxyzine", "promethazine", "paroxetine", "nortriptyline"}


def drug_interaction_checker(drug_list: list, age: int = None, egfr: float = None, systolic_bp: int = None) -> dict:
    """Simple, illustrative checks for bladder medicines. Always confirm in the BNF."""
    meds = {d.strip().lower() for d in drug_list}
    warnings = []
    acb = meds & (ANTICHOLINERGICS | OTHER_ACB)
    if len(acb) >= 2 or (acb and age and age >= 65):
        warnings.append(f"Anticholinergic burden: {', '.join(sorted(acb))}. Review cumulative burden, especially in older people.")
    if meds & {"mirabegron", "vibegron"} and systolic_bp and systolic_bp >= 160:
        warnings.append("Beta-3 agonist with raised blood pressure: mirabegron is contraindicated in severe uncontrolled hypertension; check and monitor blood pressure.")
    if "mirabegron" in meds and "digoxin" in meds:
        warnings.append("Mirabegron with digoxin: digoxin levels may rise; monitor.")
    if "nitrofurantoin" in meds and egfr is not None and egfr < 45:
        warnings.append("Nitrofurantoin with eGFR below 45: usually avoided; check current guidance.")
    return {"checked": sorted(meds), "interactions": warnings, "note": "Illustrative rules only - confirm with the BNF."}


def audit_logger(event: dict) -> dict:
    """Append an audit entry (timestamp, input hash, output summary). No raw notes are stored."""
    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "input_sha256": hashlib.sha256(event.get("input", "").encode()).hexdigest(),
             **{k: v for k, v in event.items() if k != "input"}}
    entry["audit_id"] = entry["input_sha256"][:12] + "-" + str(int(time.time() * 1000))[-6:]
    with AUDIT_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"audit_id": entry["audit_id"], "timestamp": entry["timestamp"]}


def guideline_version_checker(guideline_name: str) -> dict:
    """Version recorded in the knowledge base for a named guideline (e.g. 'NICE NG123')."""
    cards = json.loads(KB_PATH.read_text())["cards"]
    hits = {c["version"] for c in cards if guideline_name.lower() in c["source"].lower()}
    return {"guideline": guideline_name, "versions_in_index": sorted(hits) or ["not indexed"],
            "advice": "Check the publisher's website for the current version before use."}


def imaging_report_reader(report: str) -> dict:
    """Read findings from a cystoscopy or radiology report (tumour, Hunner lesion, hydronephrosis, reflux, catheter status)."""
    from .imaging import read_report
    return read_report(report)


TOOLS = {f.__name__: f for f in [fhir_patient_query, iciq_score_calculator, bpss_score_calculator,
                                  lab_result_parser, drug_interaction_checker, audit_logger, guideline_version_checker,
                                  imaging_report_reader]}
