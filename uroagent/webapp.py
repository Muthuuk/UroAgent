"""Gradio web app for UroAgent: runs in Colab (share link) and on Hugging Face Spaces."""
import json
import uuid

import gradio as gr

from . import DISCLAIMER, __version__
from .cli import EXAMPLES
from .evaluate import fairness_check, run_evaluation
from .imaging import IMAGING_DISCLAIMER, NoVisionBackend, analyse
from .patient import PATIENT_DISCLAIMER, PatientAssistant
from .scoring import iciq_oab, oleary_sant
from .tools import TOOLS

TOOL_EXAMPLES = {
    "fhir_patient_query": {"patient_id": "demo-002"},
    "iciq_score_calculator": {"frequency": 3, "nocturia": 2, "urgency": 3, "urge_leak": 2},
    "bpss_score_calculator": {"icsi_items": [4, 3, 3, 4], "icpi_items": [3, 3, 3, 4]},
    "lab_result_parser": {"report": "Urinalysis: nitrite positive; leukocytes 2+; blood negative; culture: >10^5 E. coli; creatinine 78"},
    "drug_interaction_checker": {"drug_list": ["oxybutynin", "amitriptyline", "mirabegron"], "age": 72, "systolic_bp": 165},
    "audit_logger": {"input": "example note", "event": "manual test"},
    "guideline_version_checker": {"guideline_name": "NICE NG123"},
    "imaging_report_reader": {"report": "Flexible cystoscopy: trabeculated bladder, small papillary lesion left wall. SPC in situ, encrusted."},
}


PATIENT_EXAMPLES = ["It burns when I wee and I keep needing to go to the toilet.",
                    "I keep getting sudden urges to pee and sometimes leak before I get there.",
                    "I've had a sore bladder for months and it's worse when it's full.",
                    "There's blood in my wee."]
MODALITIES = ["CT", "CT urogram", "MRI", "MRI prostate (multiparametric)", "MRI bladder", "Ultrasound", "X-ray / KUB", "Micturating cystourethrogram (contrast)", "Cystoscopy image",
              "Report only (cystoscopy or radiology)"]


def build_app(agent, patient=None, vision=None):
    patient = patient or PatientAssistant(agent)
    vision = vision or NoVisionBackend()

    def patient_chat(message, history, sid):
        if not message.strip():
            return history, sid
        r = patient.reply(message, session_id=sid)
        return (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": r["response"]}], sid

    def patient_new():
        return [], str(uuid.uuid4())[:8]

    def run_imaging(files, modality, question, consent):
        paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]
        if not paths:
            return "Please upload at least one image, DICOM file or report."
        try:
            return analyse(paths, modality=modality, question=question, backend=vision, consent=consent)["markdown"]
        except Exception as e:                 # show the problem on screen rather than an empty box
            return f"**Something went wrong:** {type(e).__name__}: {str(e)[:400]}"

    def chat(message, history, thread_id):
        if not message.strip():
            return history, "", "", thread_id
        r = agent.ask(message, thread_id=thread_id)
        history = (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": r["response"]}]
        trace = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(r["trace"])) + f"\n\nEngine: {agent.engine} · {r['latency_s']} s"
        return history, trace, r.get("soap_report", ""), thread_id

    def new_thread():
        return [], "", "", str(uuid.uuid4())[:8]

    def score(f, n, u, l, s1, s2, s3, s4, p1, p2, p3, p4):
        i = iciq_oab(f, n, u, l)
        o = oleary_sant([s1, s2, s3, s4], [p1, p2, p3, p4])
        return (f"### ICIQ-OAB structure: {i['score']}/16 ({i['band']})\n"
                f"### O'Leary-Sant: symptom index {o['icsi']}/20, problem index {o['icpi']}/16 ({o['band']})\n\n"
                "Question wording is licensed and is not reproduced here; enter item scores from the licensed questionnaires.")

    def run_tool(name, args):
        try:
            return json.dumps(TOOLS[name](**json.loads(args)), indent=2)
        except Exception as e:
            return f"Error: {e}"

    def evaluate():
        return run_evaluation(agent), fairness_check(agent)

    with gr.Blocks(title="UroAgent") as demo:
        gr.Markdown(f"# UroAgent v{__version__}\nMulti-agent decision support for overactive bladder (OAB), "
                    f"bladder pain syndrome (BPS) and urinary tract infection (UTI).\n\n**{DISCLAIMER}**")
        thread = gr.State(lambda: str(uuid.uuid4())[:8])   # a fresh thread per visitor
        psession = gr.State(lambda: str(uuid.uuid4())[:8])
        with gr.Tabs(selected="home") as tabs:
            with gr.Tab("Home", id="home"):
                gr.HTML(WELCOME_HTML)
                tiles = []
                with gr.Row(equal_height=True):
                    for i, (tab_id, label) in enumerate(HOME_TILES[:3]):
                        tiles.append((gr.Button(label, elem_classes=["uro-tile", f"uro-tile-{i + 1}"]), tab_id))
                with gr.Row(equal_height=True):
                    for i, (tab_id, label) in enumerate(HOME_TILES[3:]):
                        tiles.append((gr.Button(label, elem_classes=["uro-tile", f"uro-tile-{i + 4}"]), tab_id))
                gr.HTML('<p class="uro-foot">Research demonstrator only - synthetic data - not a medical device. '
                        'Please do not enter real patient details or upload identifiable images.</p>')
            with gr.Tab("Patients: ask about symptoms", id="patients"):
                gr.Markdown("**For people with bladder or urine symptoms.** Describe what is happening in your own words. "
                            "You will get general information and advice on where to get help, not a diagnosis.\n\n"
                            f"*{PATIENT_DISCLAIMER}* Please do not enter your name, address or other personal details.")
                pbot = gr.Chatbot(height=420, label="Chat")
                pmsg = gr.Textbox(placeholder="For example: it stings when I pee and I need to go all the time", label="Your symptoms")
                with gr.Row():
                    psend = gr.Button("Send", variant="primary")
                    pclear = gr.Button("Start again")
                gr.Examples(PATIENT_EXAMPLES, inputs=pmsg, label="Examples")
                psend.click(patient_chat, [pmsg, pbot, psession], [pbot, psession]).then(lambda: "", None, pmsg)
                pmsg.submit(patient_chat, [pmsg, pbot, psession], [pbot, psession]).then(lambda: "", None, pmsg)
                pclear.click(patient_new, None, [pbot, psession])
            with gr.Tab("Clinicians: ask UroAgent", id="clinician"):
                with gr.Row():
                    with gr.Column(scale=3):
                        bot = gr.Chatbot(height=460, label="Conversation")
                        msg = gr.Textbox(placeholder="Describe the patient (synthetic only) or ask a guideline question...", label="Clinician input")
                        with gr.Row():
                            send = gr.Button("Send", variant="primary")
                            clear = gr.Button("New conversation")
                        gr.Examples(EXAMPLES, inputs=msg, label="Try these (run them in order to see memory)")
                    with gr.Column(scale=2):
                        trace = gr.Textbox(label="Reasoning trace (every step the agent took)", lines=12)
                        soap = gr.Textbox(label="SOAP report", lines=8)
                send.click(chat, [msg, bot, thread], [bot, trace, soap, thread]).then(lambda: "", None, msg)
                msg.submit(chat, [msg, bot, thread], [bot, trace, soap, thread]).then(lambda: "", None, msg)
                clear.click(new_thread, None, [bot, trace, soap, thread])
            with gr.Tab("Clinicians: imaging and reports", id="imaging"):
                status = (f"Image model: **{vision.name}**." if getattr(vision, "available", False) else
                          "Image model: **not loaded here** - reports are still read. To describe images, use the Hugging Face Space, or in "
                          "Colab choose a T4 GPU runtime, add an HF_TOKEN secret and run all cells again." + (f" ({vision.load_error})" if getattr(vision, "load_error", "") else ""))
                gr.Markdown("Upload CT, MRI, ultrasound or contrast images (PNG, JPG or DICOM) and cystoscopy or radiology reports "
                            f"(PDF or text). {status}\n\n**{IMAGING_DISCLAIMER}**")
                with gr.Row():
                    with gr.Column():
                        files = gr.File(file_count="multiple", label="Images, DICOM files or reports",
                                        file_types=[".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".pdf", ".txt"])
                        modality = gr.Dropdown(MODALITIES, value="CT", label="Type of study")
                        question = gr.Textbox(label="Clinical question", placeholder="e.g. Is the left ureter dilated? Is the SPC balloon in the bladder? Any sign of bladder or prostate cancer?")
                        consent = gr.Checkbox(label="I confirm these files are anonymised or synthetic and contain no identifiable patient information")
                        go = gr.Button("Analyse", variant="primary")
                    with gr.Column():
                        img_out = gr.Markdown("Results appear here. The first image can take 1-2 minutes while the GPU starts.")
                go.click(run_imaging, [files, modality, question, consent], img_out)
            with gr.Tab("Score calculator", id="scores"):
                gr.Markdown("Item scores from the licensed ICIQ-OAB (0-4 each) and O'Leary-Sant questionnaires.")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**ICIQ-OAB items (0-4)**")
                        iciq = [gr.Slider(0, 4, step=1, value=v, label=l) for l, v in
                                [("Daytime frequency", 2), ("Night-time urination", 2), ("Urgency", 3), ("Urge leakage", 1)]]
                    with gr.Column():
                        gr.Markdown("**O'Leary-Sant symptom index (0-5)**")
                        icsi = [gr.Slider(0, 5, step=1, value=2, label=f"Symptom item {k}") for k in range(1, 5)]
                    with gr.Column():
                        gr.Markdown("**O'Leary-Sant problem index (0-4)**")
                        icpi = [gr.Slider(0, 4, step=1, value=2, label=f"Problem item {k}") for k in range(1, 5)]
                out = gr.Markdown()
                gr.Button("Calculate", variant="primary").click(score, iciq + icsi + icpi, out)
            with gr.Tab("Clinical tools (MCP)", id="tools"):
                gr.Markdown("The clinical tools the agent can call, also available as a Model Context Protocol server "
                            "(`python -m uroagent.mcp_server`). Synthetic patients: demo-001, demo-002, demo-003.")
                name = gr.Dropdown(list(TOOLS), value="drug_interaction_checker", label="Tool")
                args = gr.Code(value=json.dumps(TOOL_EXAMPLES["drug_interaction_checker"], indent=2), language="json", label="Input (JSON)")
                name.change(lambda n: json.dumps(TOOL_EXAMPLES[n], indent=2), name, args)
                result = gr.Code(language="json", label="Output")
                gr.Button("Run tool", variant="primary").click(run_tool, [name, args], result)
            with gr.Tab("Evaluation", id="evaluation"):
                gr.Markdown("Checks the acceptance targets from the case study on synthetic data (about 10 seconds).")
                b = gr.Button("Run evaluation", variant="primary")
                t1 = gr.Dataframe(label="Targets")
                t2 = gr.Dataframe(label="Fairness: recall by sex and age group")
                b.click(evaluate, None, [t1, t2])
            with gr.Tab("About", id="about"):
                gr.Markdown(ABOUT)
        for button, tab_id in tiles:      # each 3D tile opens its tab
            button.click(lambda t=tab_id: gr.Tabs(selected=t), None, tabs)
    _attach_style(demo)
    return demo


def _attach_style(demo):
    """Gradio 6 takes custom CSS at launch(), so wrap launch() to always include the UroAgent style."""
    original = demo.launch

    def launch(*args, **kwargs):
        kwargs["css"] = CSS + (kwargs.get("css") or "")
        return original(*args, **kwargs)

    demo.launch = launch


HOME_TILES = [
    ("patients", "🩺  Patients\nAsk about your symptoms"),
    ("clinician", "👩‍⚕️  Clinicians\nAsk UroAgent about a patient"),
    ("imaging", "🩻  Imaging and reports\nCT, MRI, cystoscopy, cancer checks"),
    ("scores", "📊  Score calculator\nICIQ-OAB and O'Leary-Sant"),
    ("tools", "🧰  Clinical tools\nMCP tools the agent uses"),
    ("evaluation", "✅  Evaluation\nAccuracy, safety and fairness"),
]

WELCOME_HTML = f"""
<div class="uro-hero">
  <div class="uro-hero-title">Welcome to the UroAgentic AI App</div>
  <div class="uro-hero-sub">AI support for bladder and urinary health - for patients with symptoms and for clinicians.
  Choose where you would like to start.</div>
  <div class="uro-hero-ver">UroAgent v{__version__}</div>
</div>
"""

CSS = """
.uro-hero, .uro-hero * { color:#fff !important; }
.uro-hero { text-align:center; padding:34px 20px 26px; margin:6px 0 18px; border-radius:22px; color:#fff;
  background:linear-gradient(120deg,#0b5fa5,#7b2ff7,#e8457c,#ff9f1c,#0b5fa5); background-size:300% 300%;
  animation:uroShift 12s ease infinite; box-shadow:0 10px 30px rgba(0,0,0,.18); }
.uro-hero-title { font-size:2.2rem; font-weight:800; letter-spacing:.3px; text-shadow:0 2px 6px rgba(0,0,0,.25); }
.uro-hero-sub { font-size:1.08rem; margin-top:10px; opacity:.95; }
.uro-hero-ver { font-size:.85rem; margin-top:10px; opacity:.8; }
@keyframes uroShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.uro-tile { min-height:118px !important; border-radius:18px !important; border:none !important; color:#fff !important;
  font-size:1.05rem !important; font-weight:700 !important; white-space:pre-line !important; line-height:1.5 !important;
  transform:translateY(0); transition:transform .15s ease, box-shadow .15s ease;
  box-shadow:0 8px 0 rgba(0,0,0,.25), 0 14px 24px rgba(0,0,0,.18) !important;
  background-size:200% 200% !important; animation:uroGlow 3.5s ease-in-out infinite; }
.uro-tile:hover { transform:translateY(-4px); box-shadow:0 12px 0 rgba(0,0,0,.25), 0 20px 30px rgba(0,0,0,.22) !important; }
.uro-tile:active { transform:translateY(6px); box-shadow:0 2px 0 rgba(0,0,0,.25), 0 6px 10px rgba(0,0,0,.18) !important; }
.uro-tile-1 { background-image:linear-gradient(135deg,#00b4d8,#0077b6,#00b4d8) !important; }
.uro-tile-2 { background-image:linear-gradient(135deg,#7b2ff7,#4361ee,#7b2ff7) !important; animation-delay:.5s; }
.uro-tile-3 { background-image:linear-gradient(135deg,#f72585,#b5179e,#f72585) !important; animation-delay:1s; }
.uro-tile-4 { background-image:linear-gradient(135deg,#ff9f1c,#f3722c,#ff9f1c) !important; animation-delay:1.5s; }
.uro-tile-5 { background-image:linear-gradient(135deg,#2a9d8f,#1b7f5b,#2a9d8f) !important; animation-delay:2s; }
.uro-tile-6 { background-image:linear-gradient(135deg,#e63946,#9d0208,#e63946) !important; animation-delay:2.5s; }
@keyframes uroGlow { 0%,100%{background-position:0% 50%; filter:brightness(1)} 50%{background-position:100% 50%; filter:brightness(1.15)} }
.uro-foot { text-align:center; font-size:.9rem; opacity:.75; margin-top:14px; }
@media (prefers-reduced-motion: reduce) { .uro-hero, .uro-tile { animation:none !important; } }
"""


ABOUT = """
**How it works.** A triage agent reads the note, extracts findings and checks safety rules. Urgent red flags
(visible haematuria, possible sepsis, retention, cauda equina) and out-of-scope or instruction-override requests are
escalated before any prediction. Otherwise a supervisor routes the request: diagnosis goes to a soft-voting ensemble
(Random Forest, XGBoost, SVM) with a 0.72 confidence gate and SHAP reasons, then to guideline retrieval and an
explanation with citations; scoring goes to the severity calculators; guideline questions go straight to retrieval.
Conversation memory is kept per thread in SQLite, and every turn is written to an audit log (a hash of the input, never
the note itself).

**Limits.** Synthetic training data; guideline cards are short paraphrased summaries, not guideline text, and may be
out of date; no real EHR connection (the FHIR tool returns synthetic records). Not a medical device. Any clinical use
would need a clinical safety case (DCB0129/DCB0160), MHRA conformity assessment and validation on real data.

Design: *AI Agent for Urological Prediction Models: A Full Modern Software Engineering Approach* (Ramachandran, 2026).
"""
