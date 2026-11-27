"""UroAgent: the multi-agent workflow (case study sections 4.3 and 4.4).

Graph:  triage --(oob or urgent red flag)--> escalate
              --(diagnosis)--> classify -> retrieve -> explain -> respond
              --(guideline)--> retrieve -> explain -> respond
              --(scoring)----> score -> respond

The graph is compiled with LangGraph when it is installed; otherwise an equivalent
built-in runner executes the same nodes, so the demo works anywhere. No language model
or API key is needed. An optional language model can reword the final explanation
(see explain_with_llm), but it can never change predictions, scores or safety rules.
"""
import os
import re
import time

from . import DISCLAIMER, __version__, safety, tools
from .classifier import EnsembleClassifier
from .memory import SQLiteCheckpointer
from .ner import describe, extract_entities
from .rag import GuidelineRetriever
from .scoring import estimate_from_entities
from .state import UroAgentState

SCORING_WORDS = ["score", "iciq", "o'leary", "oleary", "severity", "how severe", "questionnaire"]
GUIDELINE_WORDS = ["guideline", "nice", "eau", "aua", "what should", "first-line", "first line", "management",
                   "treat", "how do i", "recommend", "next step", "what do i do", "what next", "advice", "manage", "?"]
URO_TERMS = ["urin", "bladder", "uti", "urgen", "frequen", "nocturia", "leak", "incontinen", "cystitis", "dysuria",
             "burning", "pelvic", "dipstick", "nitrite", "culture", "mirabegron", "antimuscarinic", "oab", "bps",
             "interstitial", "pee", "wee", "void", "hunner", "botulinum", "toilet", "haematuria", "hematuria"]
URGENT = {"visible haematuria", "possible pyelonephritis or sepsis", "urinary retention", "possible cauda equina"}


def _trace(state, msg):
    return (state.get("trace") or []) + [msg]


class UroAgent:
    def __init__(self, memory_path=":memory:", train_size=1500, use_langgraph=True):
        t0 = time.time()
        self.classifier = EnsembleClassifier().fit(n=train_size)
        self.retriever = GuidelineRetriever()
        self.memory = SQLiteCheckpointer(memory_path)
        self.engine = "built-in runner"
        self.graph = self._build_langgraph() if use_langgraph else None
        self.startup_seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ nodes
    def triage_node(self, state: UroAgentState) -> dict:
        note = state["clinical_note"]
        low = note.lower()
        ents = extract_entities(note)
        findings = describe(ents)
        flags = safety.red_flags(note)
        update = {"entities": ents, "red_flags": [f"{n}: {a}" for n, a in flags], "escalate": False}
        if safety.detect_injection(note):
            qt, why = "oob", "the message tries to override the agent's safety instructions"
        elif safety.out_of_scope_reason(note):
            qt, why = "oob", f"it concerns {safety.out_of_scope_reason(note)}, which is outside this agent's urology scope"
        elif state.get("clinical_context") and state["clinical_context"].get("prediction") and len(findings) < 2 and (
                any(w in low for w in GUIDELINE_WORDS) or low.strip().endswith("?")):
            qt, why = "guideline", f"it is a follow-up question about the earlier {state['clinical_context']['prediction']} assessment in this conversation"
        elif any(w in low for w in SCORING_WORDS):
            qt, why = "scoring", "a symptom severity score was requested"
        elif len(findings) >= 2 and not low.strip().endswith("?"):
            qt, why = "diagnosis", f"{len(findings)} urological findings were described"
        elif any(w in low for w in GUIDELINE_WORDS) and any(w in low for w in URO_TERMS):
            qt, why = "guideline", "a guidance question was asked"
        elif len(findings) >= 1 and any(w in low for w in URO_TERMS):
            qt, why = "diagnosis", "urological symptoms were described"
        else:
            qt, why = "oob", "no urology question or findings were found"
        update["query_type"] = qt
        update["trace"] = _trace(state, f"Triage: classified as '{qt}' because {why}." +
                                 (f" Findings: {', '.join(findings)}." if findings else ""))
        if any(n in URGENT for n, _ in flags):
            update["trace"].append("Safety: urgent red flag found - routing to escalation before any prediction.")
        return update

    def route(self, state: UroAgentState) -> str:
        if state["query_type"] == "oob" or any(f.split(":")[0] in URGENT for f in state.get("red_flags", [])):
            return "escalate"
        return {"diagnosis": "classify", "scoring": "score", "guideline": "retrieve"}[state["query_type"]]

    def classify_node(self, state: UroAgentState) -> dict:
        r = self.classifier.predict(state["entities"])
        msg = (f"Classifier: soft-voting ensemble (RF + XGBoost + SVM) gives {r['top_class']} "
               f"with probability {r['confidence']:.2f}")
        msg += "." if r["prediction"] != "Indeterminate" else f", below the 0.72 confidence gate, so the result is Indeterminate."
        return {**r, "trace": _trace(state, msg)}

    def score_node(self, state: UroAgentState) -> dict:
        s = estimate_from_entities(state["entities"])
        i, o = s["iciq_oab"], s["oleary_sant"]
        return {"iciq_oab": i, "oleary_sant": o,
                "trace": _trace(state, f"Scoring: estimated ICIQ-OAB {i['score']}/16 ({i['band']}); "
                                       f"O'Leary-Sant ICSI {o['icsi']}/20, ICPI {o['icpi']}/16.")}

    def retrieve_node(self, state: UroAgentState) -> dict:
        cond = state.get("prediction") if state.get("prediction") in ("OAB", "BPS", "UTI") else state.get("top_class")
        if not cond and state.get("clinical_context"):
            cond = state["clinical_context"].get("prediction")
        findings = describe(state.get("entities") or {}) or (state.get("clinical_context") or {}).get("findings", [])
        query = state["clinical_note"]
        if len(describe(state.get("entities") or {})) == 0 and cond:
            query = f"{query} {cond} " + " ".join(findings)
        hits = self.retriever.search(query, condition=cond, findings=findings, k=5)
        return {"retrieved_context": hits, "guideline_citations": [h["citation"] for h in hits],
                "trace": _trace(state, f"Retrieval: hybrid BM25 + vector search with rank fusion returned "
                                       f"{len(hits)} guideline passages" + (f" focused on {cond}." if cond else "."))}

    def explain_node(self, state: UroAgentState) -> dict:
        lines = []
        if state.get("prediction"):
            p = state["prediction"]
            probs = ", ".join(f"{k} {v:.2f}" for k, v in state["probabilities"].items())
            if p == "Indeterminate":
                lines.append(f"**Assessment:** Indeterminate (closest: {state['top_class']}, probability {state['confidence']:.2f}; all: {probs}). "
                             "The pattern is not clear enough for a confident suggestion; specialist review is advised.")
            else:
                lines.append(f"**Most likely pattern:** {p} (probability {state['confidence']:.2f}; all: {probs}).")
            if state.get("feature_importances"):
                ups = [k for k, v in state["feature_importances"].items() if v > 0][:3]
                downs = [k for k, v in state["feature_importances"].items() if v < 0][:2]
                why = []
                if ups: why.append("pointing towards it: " + ", ".join(ups))
                if downs: why.append("pointing away: " + ", ".join(downs))
                lines.append("**Why (SHAP):** " + "; ".join(why) + ".")
        ctx = state.get("retrieved_context") or []
        prev = (state.get("clinical_context") or {}).get("prediction")
        if not state.get("prediction") and prev:
            lines.append(f"Following on from the earlier assessment in this conversation (most likely pattern: {prev}).")
        if ctx:
            lines.append("**What the guidance says (paraphrased summaries):**")
            for n, h in enumerate(ctx[:4], 1):
                lines.append(f"{n}. {h['action']} [{n}]")
        return {"explanation": "\n".join(lines), "trace": _trace(state, "Explanation: combined prediction, SHAP reasons and cited guidance.")}

    def respond_node(self, state: UroAgentState) -> dict:
        out = [f"*{DISCLAIMER}*", ""]
        cautions = [f for f in state.get("red_flags", []) if f.split(":")[0] not in URGENT]
        if cautions:
            out += ["**Caution:** " + "; ".join(c.split(": ", 1)[1] for c in cautions), ""]
        if state.get("explanation"):
            out.append(state["explanation"])
        if state.get("iciq_oab"):
            i, o = state["iciq_oab"], state["oleary_sant"]
            out.append(f"**Estimated scores (from the note, approximate):** ICIQ-OAB {i['score']}/16 ({i['band']}); "
                       f"O'Leary-Sant symptom index {o['icsi']}/20, problem index {o['icpi']}/16. "
                       "Use the licensed questionnaires for a formal score.")
        if state.get("guideline_citations"):
            out += ["", "**Sources:**"] + [f"[{n}] {c}" for n, c in enumerate(state["guideline_citations"][:4], 1)]
        out += ["", "**Clinician sign-off required.** This is advisory output; you can override or dismiss it."]
        response = "\n".join(out)
        soap = self._soap(state)
        audit = tools.audit_logger({"input": state["clinical_note"], "query_type": state["query_type"],
                                    "prediction": state.get("prediction"), "confidence": state.get("confidence"),
                                    "citations": len(state.get("guideline_citations") or []), "model_version": f"uroagent-{__version__}"})
        return {"response": response, "soap_report": soap,
                "trace": _trace(state, f"Response: written with {len(state.get('guideline_citations') or [])} citations; audit id {audit['audit_id']}.")}

    def escalate_node(self, state: UroAgentState) -> dict:
        urgent = [f for f in state.get("red_flags", []) if f.split(":")[0] in URGENT]
        if urgent:
            body = ["**Urgent safety concern - please act on this before anything else:**"] + [f"- {f.split(': ', 1)[1]}" for f in urgent]
            body.append("No diagnostic suggestion is given while a red flag is present.")
        else:
            reason = state["trace"][-1] if state.get("trace") else ""
            body = ["**Outside this agent's scope.** UroAgent only supports questions about overactive bladder, "
                    "bladder pain syndrome and urinary tract infection in adults.",
                    "Please use the appropriate clinical service or pathway." ]
            if "override" in reason:
                body.append("Requests to change the agent's instructions are declined and logged.")
        audit = tools.audit_logger({"input": state["clinical_note"], "query_type": state["query_type"], "escalated": True})
        return {"escalate": True, "response": f"*{DISCLAIMER}*\n\n" + "\n".join(body),
                "trace": _trace(state, f"Escalation: safe holding response given; audit id {audit['audit_id']}.")}

    @staticmethod
    def _soap(state):
        e = state.get("entities") or {}
        o = [f"Age {int(e.get('age', 0))}", f"day frequency {int(e.get('day_frequency', 0))}", f"nocturia {int(e.get('nocturia', 0))}"]
        if e.get("nitrite") or e.get("leukocytes"):
            o.append("dipstick: " + ", ".join(x for x, k in (("nitrite +", "nitrite"), ("leukocytes +", "leukocytes")) if e.get(k)))
        a = state.get("prediction") or "Not assessed"
        if state.get("confidence"):
            a += f" (probability {state['confidence']:.2f})"
        plan = [h["action"] for h in (state.get("retrieved_context") or [])[:3]]
        return ("S: " + state["clinical_note"] + "\nO: " + "; ".join(o) + "\nA: " + a +
                "\nP: " + (" ".join(plan) if plan else "Clinician to decide.") + " Safety-net advice given. Clinician sign-off required.")

    # ------------------------------------------------------------------ graph
    def _build_langgraph(self):
        try:
            from langgraph.graph import StateGraph, END
        except Exception:
            return None
        g = StateGraph(UroAgentState)
        for name in ("triage", "classify", "score", "retrieve", "explain", "respond", "escalate"):
            g.add_node(name, getattr(self, f"{name}_node"))
        g.set_entry_point("triage")
        g.add_conditional_edges("triage", self.route, {"escalate": "escalate", "classify": "classify", "score": "score", "retrieve": "retrieve"})
        g.add_edge("classify", "retrieve")
        g.add_edge("retrieve", "explain")
        g.add_edge("explain", "respond")
        g.add_edge("score", "respond")
        g.add_edge("respond", END)
        g.add_edge("escalate", END)
        self.engine = "LangGraph StateGraph"
        return g.compile()

    def _run_builtin(self, state):
        nxt = {"classify": "retrieve", "retrieve": "explain", "explain": "respond", "score": "respond"}
        state.update(self.triage_node(state))
        node = self.route(state)
        while node:
            state.update(getattr(self, f"{node}_node")(state))
            node = nxt.get(node)
        return state

    # ------------------------------------------------------------------ public API
    def ask(self, note: str, thread_id: str = "default", patient_id: str = None) -> dict:
        """Run one turn. Returns the final state (response, trace, scores, citations...)."""
        t0 = time.time()
        memory = self.memory.load(thread_id)
        messages = (memory.get("messages") or []) + [{"role": "user", "content": note}]
        state: UroAgentState = {"clinical_note": note, "thread_id": thread_id, "patient_id": patient_id,
                                "messages": messages, "trace": [],
                                "clinical_context": {"prediction": memory.get("prediction"),
                                                     "findings": describe(memory.get("entities") or {})} if memory else None}
        if patient_id:
            bundle = tools.fhir_patient_query(patient_id)
            state["trace"] = [f"FHIR tool: loaded synthetic record for {patient_id} ({bundle['total']} resources)."]
        result = dict(self.graph.invoke(state)) if self.graph else self._run_builtin(state)
        result["messages"] = messages + [{"role": "assistant", "content": result.get("response", "")}]
        if result.get("query_type") == "diagnosis":
            self.memory.save(thread_id, result)
        else:
            self.memory.save(thread_id, {**memory, "messages": result["messages"]})
        result["latency_s"] = round(time.time() - t0, 3)
        return result


def explain_with_llm(state: dict, model: str = None) -> str:
    """Optional: reword the explanation in plain English with Claude (needs ANTHROPIC_API_KEY).
    The model only rewrites; it is told not to add facts, numbers or advice."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return state.get("response", "")
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    prompt = ("Rewrite the following clinical decision-support output in clear plain English for a clinician. "
              "Do not add any new facts, numbers, diagnoses or advice, keep every citation marker, and keep the "
              "disclaimer and sign-off lines.\n\n" + state.get("response", ""))
    msg = client.messages.create(model=model or os.environ.get("UROAGENT_LLM_MODEL", "claude-sonnet-5-5"),
                                 max_tokens=800, messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text
