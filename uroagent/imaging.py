"""Imaging and report analysis for clinicians (research demonstrator).

Inputs: images (PNG/JPG), DICOM slices (.dcm), and cystoscopy or radiology reports
(PDF or text). Images are described by a medical vision-language model (Google
MedGemma) when it is available, using a structured urology checklist. Reports are
read with transparent rules. Every result is labelled as an AI-generated draft that
must be reviewed by a radiologist or urologist.

Privacy: only anonymised or synthetic images may be used, especially on the public
demo. DICOM files are stripped of identifying tags before analysis, and nothing is
stored after the request.
"""
import io
import os
import re
from dataclasses import dataclass, field

from PIL import Image

IMAGING_DISCLAIMER = ("AI-generated draft for research and teaching only - not a radiology or cystoscopy report. "
                      "It can miss or invent findings and must be reviewed by a radiologist or urologist. "
                      "Use anonymised or synthetic images only.")

CHECKLIST = [
    "image type, body region and plane, and whether the image quality is adequate",
    "kidneys: hydronephrosis (dilated collecting system), stones, masses",
    "ureters: dilatation (hydroureter) and its level, stones, strictures or other cause of obstruction",
    "bladder: wall thickening or trabeculation, diverticula, stones, filling defects or masses",
    "possible bladder cancer: a focal or irregular wall thickening, a mass projecting into the bladder, a filling defect, "
    "spread into the fat around the bladder; on MRI, say whether the muscle layer looks breached (VI-RADS style)",
    "prostate: size (enlarged or not), any focal lesion and its zone (peripheral or transition); on prostate MRI, "
    "a PI-RADS style impression, spread outside the prostate capsule, seminal vesicle involvement",
    "spread of cancer: enlarged pelvic or abdominal lymph nodes, bone lesions (sclerotic or lytic), other organs",
    "contrast reflux into the ureters, only if this is a contrast study such as a micturating cystourethrogram",
    "catheters and tubes: suprapubic or urethral catheter, ureteric stent, nephrostomy - where the tip or balloon lies and "
    "whether it looks correctly placed, displaced, kinked or encrusted",
    "any other abnormality and any urgent finding",
]

PROMPT = ("You are assisting a urologist with an anonymised {modality} image. Clinical question: {question}\n"
          "Describe only what is visible. Work through this checklist and answer each item as 'seen', 'not seen' or "
          "'cannot assess from this image', with a short reason:\n" + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(CHECKLIST)) +
          "\nThen give a one-line overall impression and say what further imaging or review would help. "
          "Be cautious: if unsure, say so. Do not invent patient details.")

DICOM_IDENTIFYING = ["PatientName", "PatientID", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
                     "OtherPatientIDs", "OtherPatientNames", "ReferringPhysicianName", "InstitutionName",
                     "InstitutionAddress", "AccessionNumber", "StudyID", "PerformingPhysicianName", "OperatorsName"]


@dataclass
class LoadedFile:
    name: str
    kind: str                     # 'image' | 'report'
    image: Image.Image = None
    text: str = ""
    metadata: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def load_file(path: str) -> LoadedFile:
    """Read an image, DICOM file, PDF or text file into a common structure."""
    name = os.path.basename(path)
    low = name.lower()
    if low.endswith((".txt", ".md")):
        return LoadedFile(name, "report", text=open(path, encoding="utf-8", errors="ignore").read())
    if low.endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        return LoadedFile(name, "report", text=text)
    if low.endswith((".dcm", ".dicom")) or _looks_like_dicom(path):
        return _load_dicom(path, name)
    img = Image.open(path).convert("RGB")
    return LoadedFile(name, "image", image=img, metadata={"size": img.size})


def _looks_like_dicom(path):
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _load_dicom(path, name):
    import numpy as np
    import pydicom
    ds = pydicom.dcmread(path)
    removed = [t for t in DICOM_IDENTIFYING if t in ds and str(ds.get(t, "")).strip()]
    for t in DICOM_IDENTIFYING:
        if t in ds:
            delattr(ds, t)
    meta = {k: str(ds.get(k, "")) for k in ("Modality", "BodyPartExamined", "StudyDescription", "SeriesDescription",
                                             "SliceThickness", "PixelSpacing") if ds.get(k, None) not in (None, "")}
    warnings = []
    if removed:
        warnings.append(f"Identifying DICOM fields were present and have been removed before analysis: {', '.join(removed)}. "
                        "Only anonymised images should be uploaded.")
    if str(ds.get("BurnedInAnnotation", "")).upper() == "YES":
        warnings.append("The file says it has text burned into the image (which may include identifiers). Do not upload such images.")
    img = None
    if "PixelData" in ds:
        arr = ds.pixel_array.astype("float32")
        if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
            arr = arr[arr.shape[0] // 2]              # middle slice of a multi-frame series
        slope, intercept = float(ds.get("RescaleSlope", 1) or 1), float(ds.get("RescaleIntercept", 0) or 0)
        arr = arr * slope + intercept
        centre, width = ds.get("WindowCenter", None), ds.get("WindowWidth", None)
        if centre is not None and width is not None:
            c = float(centre[0] if hasattr(centre, "__len__") and not isinstance(centre, str) else centre)
            w = float(width[0] if hasattr(width, "__len__") and not isinstance(width, str) else width)
            lo, hi = c - w / 2, c + w / 2
        else:
            lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        img = Image.fromarray(arr.astype("uint8")).convert("RGB")
    return LoadedFile(name, "image", image=img, metadata=meta, warnings=warnings)


# ---------------------------------------------------------------- report reading (rules)
_NOT_PROSTATE = r"(?<!prostate )(?<!prostatic )(?<!adeno)"
REPORT_RULES = [
    ("Possible bladder tumour", _NOT_PROSTATE + r"(papillary|tumou?r|growth|mass|suspicious lesion|sessile lesion|carcinoma)", "urgent",
     "Suspicious bladder lesion: needs urgent urology review (suspected cancer pathway)."),
    ("Possible muscle-invasive bladder cancer", r"vi-?rads\W*(score\W*)?[45]\b|muscle[- ]invasi|detrusor (muscle )?invasion|"
     r"perivesical (fat )?(invasion|extension|stranding)|extravesical", "urgent",
     "Features of muscle-invasive or locally advanced bladder cancer: urgent urology and oncology (MDT) review for staging."),
    ("Upper tract filling defect", r"filling defect[^.]{0,40}(ureter|renal pelvis|collecting system|calyx|calyces)|"
     r"(ureter|renal pelvis|collecting system)[^.]{0,40}filling defect", "urgent",
     "Filling defect in the upper urinary tract: possible upper tract urothelial cancer, urgent urology review."),
    ("Suspected prostate cancer (PI-RADS 4-5)", r"pi-?rads\W*(v?2(\.1)?\W*)?(score\W*|category\W*)?(of\W*)?[45]\b|likert\W*(score\W*)?[45]\b", "urgent",
     "Lesion likely to be clinically significant prostate cancer: urgent urology review and targeted biopsy (suspected cancer pathway)."),
    ("Equivocal prostate lesion (PI-RADS 3)", r"pi-?rads\W*(v?2(\.1)?\W*)?(score\W*|category\W*)?(of\W*)?3\b|likert\W*(score\W*)?3\b", "review",
     "Equivocal prostate lesion: discuss biopsy using PSA density, risk factors and patient preference."),
    ("Low suspicion prostate MRI (PI-RADS 1-2)", r"pi-?rads\W*(v?2(\.1)?\W*)?(score\W*|category\W*)?(of\W*)?[12]\b", "info",
     "Low suspicion prostate MRI: biopsy may be avoided; follow local PSA surveillance advice."),
    ("Prostate cancer described", r"prostat\w* (adeno)?carcinoma|prostate cancer|adenocarcinoma of the prostate|gleason|isup grade", "urgent",
     "Prostate cancer described: ensure urology/oncology MDT review and staging."),
    ("Spread outside the prostate", r"extra-?prostatic (extension|spread)|capsular (breach|extension)|seminal vesicle (invasion|involvement)", "urgent",
     "Possible locally advanced prostate cancer: urgent MDT review and staging."),
    ("Enlarged lymph nodes", r"lymphadenopathy|enlarged[^.]{0,20}(lymph )?nodes?|nodal (disease|metasta)", "review",
     "Enlarged lymph nodes: may indicate spread of cancer; correlate with staging."),
    ("Possible bone metastases", r"(bone|bony|osseous|skeletal|sclerotic|lytic)[^.]{0,20}(metasta|deposit|lesion)", "urgent",
     "Possible bone spread: urgent oncology review; consider bone scan or PSMA PET as appropriate."),
    ("Enlarged prostate", r"enlarged prostate|prostatomegaly|benign prostatic|prostat\w* (hyperplasia|enlargement)|median lobe", "routine",
     "Prostate enlargement described: consistent with benign enlargement unless other features are present."),
    ("Carcinoma in situ suspected", r"\bcis\b|velvety|carcinoma in situ", "urgent", "Possible carcinoma in situ: urgent urology review and biopsy."),
    ("Hunner lesion", r"hunner", "routine", "Hunner lesion described: supports bladder pain syndrome with Hunner lesions; consider lesion-directed treatment."),
    ("Bladder inflammation", r"cystitis|erythema|inflam|oedema|edema|glomerulation", "routine", "Inflammatory change described."),
    ("Trabeculation", r"trabecula", "routine", "Trabeculation suggests long-standing bladder outlet obstruction or overactivity."),
    ("Diverticulum", r"diverticul", "routine", "Bladder diverticulum described."),
    ("Bladder stone", r"stone|calcul", "routine", "Bladder stone described."),
    ("Urethral stricture", r"stricture", "routine", "Urethral stricture described."),
    ("Ureteric orifice abnormality", r"(ureteric|ureteral) orifices? (abnormal|gaping|golf[- ]hole|ectopic)|golf[- ]hole", "routine",
     "Abnormal ureteric orifice (for example golf-hole) can be associated with reflux."),
    ("Hydronephrosis or hydroureter", r"hydroneph|hydroureter|dilat\w* (ureter|collecting system|pelvicalyceal)", "review",
     "Dilated upper tract described: check for obstruction and kidney function."),
    ("Vesicoureteric reflux", r"reflux", "review", "Reflux described: check grade and the type of study it was seen on."),
    ("SPC catheter present", r"suprapubic catheter|\bspc\b", "info", "Suprapubic catheter noted."),
    ("Catheter problem", r"encrust|blocked|blockage|bypass|displac|migrat|kink|not draining|haematuria around", "review",
     "Catheter problem described (encrustation, blockage, bypassing or displacement): review catheter care and consider change."),
    ("Catheter satisfactory", r"(catheter|spc)[^.]{0,40}(satisfactory|good position|well positioned|draining well|in situ and draining)", "info",
     "Catheter described as satisfactory."),
]


def read_report(text: str) -> dict:
    """Rule-based extraction of findings from a cystoscopy or radiology report."""
    from .ner import remove_negated
    clean = remove_negated(" " + text.lower().replace("\n", " ") + " ")
    findings = [{"finding": n, "priority": p, "note": note} for n, rx, p, note in REPORT_RULES if re.search(rx, clean)]
    normal = bool(re.search(r"\bnormal (cystoscopy|bladder|study)\b|no abnormality", text.lower())) and not any(
        f["priority"] in ("urgent", "review") for f in findings)
    order = {"urgent": 0, "review": 1, "routine": 2, "info": 3}
    findings.sort(key=lambda f: order[f["priority"]])
    return {"findings": findings, "normal_statement": normal,
            "urgent": [f for f in findings if f["priority"] == "urgent"]}


# ---------------------------------------------------------------- image model
class VisionBackend:
    name = "none"
    available = False

    def describe(self, image, modality, question):
        raise NotImplementedError


class NoVisionBackend(VisionBackend):
    """Used when no image model is loaded (for example in Colab without a GPU)."""
    name = "not loaded"

    def describe(self, image, modality, question):
        return ("The image model is not loaded in this environment, so the picture itself was not analysed. "
                "On the Hugging Face Space (or Colab with a GPU and access to MedGemma) the image is described "
                "against the urology checklist.")


class MedGemmaBackend(VisionBackend):
    """Google MedGemma (image-text-to-text). Needs a GPU and accepted model terms on Hugging Face."""
    name = "MedGemma"

    def __init__(self, model_id=None, max_new_tokens=450):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model_id = model_id or os.environ.get("UROAGENT_VISION_MODEL", "google/medgemma-4b-it")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        on_zerogpu = bool(os.environ.get("SPACES_ZERO_GPU"))   # Hugging Face ZeroGPU attaches the GPU per request
        device = "cuda" if (torch.cuda.is_available() or on_zerogpu) else "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(device)
        self.max_new_tokens = max_new_tokens
        self.available = True

    def describe(self, image, modality, question):
        import torch
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist and urologist."}]},
                    {"role": "user", "content": [{"type": "text", "text": PROMPT.format(modality=modality, question=question or "general review")},
                                                 {"type": "image", "image": image}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)
        n = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][n:], skip_special_tokens=True).strip()


def load_vision_backend():
    """Load MedGemma if requested (UROAGENT_LOAD_VISION=1) and possible; otherwise a placeholder."""
    if os.environ.get("UROAGENT_LOAD_VISION", "0") != "1":
        return NoVisionBackend()
    try:
        return MedGemmaBackend()
    except Exception as e:                     # missing GPU, packages or model access
        b = NoVisionBackend()
        b.load_error = f"{type(e).__name__}: {e}"[:300]
        return b


# ---------------------------------------------------------------- analysis entry point
URGENT_IMAGE_WORDS = r"tumou?r|mass|malignan|perforat|free air|abscess|obstruct|severe hydroneph|displaced catheter|catheter (is )?outside"


def analyse(paths, modality="CT", question="", backend=None, consent=False) -> dict:
    """Analyse uploaded files. Refuses unless the user confirms the files are anonymised or synthetic."""
    if not consent:
        return {"ok": False, "markdown": "Please confirm that the files are **anonymised or synthetic** before analysis. "
                                         "Real, identifiable patient images must not be uploaded to this demo."}
    backend = backend or NoVisionBackend()
    sections, flags = [f"*{IMAGING_DISCLAIMER}*", ""], []
    for path in paths or []:
        try:
            f = load_file(path)
        except Exception as e:
            sections.append(f"**{os.path.basename(path)}:** could not be read ({type(e).__name__}).")
            continue
        sections.append(f"### {f.name}")
        for w in f.warnings:
            sections += [f"> ⚠️ {w}", ""]
        if f.metadata:
            sections += ["Details: " + "; ".join(f"{k} {v}" for k, v in f.metadata.items()), ""]
        if f.kind == "report":
            r = read_report(f.text)
            if not r["findings"]:
                sections.append("No recognised findings in the report text." + (" The report describes a normal study." if r["normal_statement"] else ""))
            for x in r["findings"]:
                icon = {"urgent": "🔴", "review": "🟠", "routine": "🔵", "info": "⚪"}[x["priority"]]
                sections.append(f"- {icon} **{x['finding']}** ({x['priority']}): {x['note']}")
            flags += [x["note"] for x in r["urgent"]]
        elif f.image is not None:
            try:
                text = backend.describe(f.image, modality, question)
            except Exception as e:             # GPU quota, time-out or model error: show it instead of failing silently
                sections.append(f"**The image model could not analyse this image:** {type(e).__name__}: {str(e)[:300]}")
                sections.append("")
                continue
            sections.append(f"**Image description ({backend.name}):**\n\n{text}")
            if re.search(URGENT_IMAGE_WORDS, text.lower()):
                flags.append(f"{f.name}: the image description mentions a potentially urgent finding - please review promptly.")
        sections.append("")
    if flags:
        sections.insert(2, "**Needs prompt clinical review:**\n" + "\n".join(f"- {x}" for x in flags) + "\n")
    sections.append("**Clinician sign-off required.** Compare with the formal radiology report and the clinical picture.")
    return {"ok": True, "markdown": "\n".join(sections), "flags": flags, "backend": backend.name}
