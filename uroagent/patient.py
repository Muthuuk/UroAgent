"""Patient assistant: a small conversational layer for people describing their own
bladder symptoms.

It uses the same finding extraction and safety rules as the clinician agent, but it
never shows a diagnosis, probability or score. It asks a few follow-up questions
(slot filling), then gives plain-English information, self-care tips and clear
signposting: 999, NHS 111, GP, pharmacist or sexual health service. Research
demonstrator only.
"""
import re

from . import safety
from .ner import extract_entities, remove_negated

PATIENT_DISCLAIMER = ("This is a research demonstration, not a medical service. It cannot diagnose you. "
                      "If you are worried, contact your GP, a pharmacist or NHS 111. In an emergency call 999.")

QUESTIONS = {
    "duration": "How long have you had these symptoms (for example 2 days, 3 weeks, 6 months)?",
    "burning": "Does it burn or sting when you pass urine?",
    "fever": "Do you have a high temperature, shivers, or pain in your back or side below the ribs?",
    "blood": "Have you seen any blood in your urine, or is it pink, red or brown?",
    "pain": "Do you have pain or pressure low in your tummy, and does it get worse as your bladder fills?",
    "who": "To give the right advice, are you a man or a woman, roughly how old are you, and could you be pregnant?",
}

SLOT_PATTERNS = {
    "duration": r"\d+\s*(day|days|week|weeks|month|months|year|years)|yesterday|today|this morning",
    "burning": r"burn|sting|dysuria|hurts? when|pain when",
    "fever": r"temperature|fever|shiver|rigor|loin|flank|side|back",
    "blood": r"blood|pink|red|brown|haematuria|hematuria",
    "pain": r"pain|pressure|ache|tummy|pelvic|bladder",
    "who": r"\b(man|woman|male|female|he|she|pregnan|\d{2}\s*(years|yrs|year-old|year old))",
}

URGENT_999 = {"possible cauda equina"}
URGENT_SAME_DAY = {"possible pyelonephritis or sepsis", "urinary retention"}


class PatientAssistant:
    """Keeps a short-term record of what the patient has said in each conversation."""

    def __init__(self, agent=None):
        self.agent = agent            # optional: reuse the clinician agent's classifier internally
        self.sessions = {}

    def _slots(self, text):
        low = text.lower()
        return {k for k, p in SLOT_PATTERNS.items() if re.search(p, low)}

    def reply(self, message: str, session_id: str = "patient") -> dict:
        s = self.sessions.setdefault(session_id, {"text": "", "asked": set(), "turns": 0})
        s["text"] += " " + message
        s["turns"] += 1
        text = s["text"]
        low = text.lower()

        if safety.detect_injection(message):
            return self._out("I can only help with questions about bladder and urine symptoms.", "declined")
        if any(w in low for w in ("suicid", "self-harm", "self harm", "kill myself")):
            return self._out("I'm really sorry you're feeling like this. Please talk to someone now: call **Samaritans on 116 123** "
                             "(free, any time), contact your GP or NHS 111, or call **999** if you are in danger.", "crisis")

        flags = {n for n, _ in safety.red_flags(text)}
        if flags & URGENT_999:
            return self._out("**Please call 999 or go to A&E now.** Numbness around your bottom or genitals, new weakness in your legs, "
                             "or problems controlling your bladder or bowels can be signs of a serious problem with the nerves in your back "
                             "that needs emergency treatment.", "emergency")
        if flags & URGENT_SAME_DAY:
            msg = ["**Please get medical help today.**"]
            if "urinary retention" in flags:
                msg.append("Not being able to pass urine, especially with a painful swollen tummy, needs to be seen urgently: "
                           "call NHS 111 or go to A&E or an urgent treatment centre.")
            if "possible pyelonephritis or sepsis" in flags:
                msg.append("A high temperature, shivering, being sick, or pain in your back or side below the ribs can mean the infection "
                           "has spread to the kidneys. Contact your GP urgently or call NHS 111 now. If you feel very unwell, confused or "
                           "drowsy, call 999.")
            return self._out(" ".join(msg), "urgent")
        if "visible haematuria" in flags:
            return self._out("**Please book an appointment with your GP soon (within the next few days).** Blood in the urine should "
                             "always be checked by a doctor, even if it has gone away. If you also have a high temperature or "
                             "are passing clots and struggling to pass urine, call NHS 111 today.", "see GP soon")

        missing = [k for k in QUESTIONS if k not in self._slots(text) and k not in s["asked"]]
        if s["turns"] <= 2 and missing:
            ask = missing[:2]
            s["asked"].update(ask)
            intro = "Thank you for telling me. " if s["turns"] == 1 else "Thanks. "
            return self._out(intro + "A couple of questions so I can point you in the right direction:\n\n" +
                             "\n".join(f"- {QUESTIONS[k]}" for k in ask), "asking")
        return self._out(self._advice(text), "advice")

    def _advice(self, text):
        e = extract_entities(text)
        low = remove_negated(" " + text.lower() + " ")
        male = not e["female"]
        pregnant = "pregnan" in low
        older = e["age"] >= 65
        parts = []
        infection_like = e["dysuria"] or e["cloudy_urine"] or e["acute_onset"]
        if infection_like:
            parts.append("Burning or stinging when you pass urine, cloudy or smelly urine, and needing to go often, especially when it "
                         "starts suddenly, are common signs of a **urine infection** (cystitis), but other things can cause them too.")
            if pregnant or male or older:
                who = "pregnant" if pregnant else ("a man" if male else "aged 65 or over")
                parts.append(f"Because you are {who}, please **contact your GP practice or NHS 111 today**: a urine test is "
                             "usually needed before any treatment.")
            elif e["recurrent_uti"]:
                parts.append("As this keeps coming back, please **see your GP** so they can test your urine and look for a cause.")
            else:
                parts.append("If you are a woman aged 16 to 64 and not pregnant, a **pharmacist** can often help with simple urine "
                             "infections (in England through the NHS Pharmacy First service). Otherwise, contact your GP practice.")
        if e["suprapubic_pain"] and e["duration_weeks"] >= 6 and not e["acute_onset"]:
            parts.append("Bladder pain or pressure that has lasted for weeks or months should be checked by your **GP**. There are "
                         "several possible causes and treatments that can help, so it is worth getting it looked at.")
        if (e["urgency"] or e["urge_incontinence"] or e["day_frequency"] >= 10 or e["nocturia"] >= 2) and not infection_like:
            parts.append("Sudden urges to go, going very often, getting up at night or small leaks are very common and often improve "
                         "with simple changes. Your **GP** can help and may suggest bladder training.")
            parts.append("Things you can try now: cut down on caffeine (tea, coffee, cola) and fizzy drinks, drink normal amounts "
                         "(about 6 to 8 glasses a day) rather than too much or too little, and keep a simple diary of when and how "
                         "much you go for 3 days to show your GP.")
        if not parts:
            parts.append("I couldn't match your symptoms to common bladder problems. Please speak to your **GP** or a pharmacist, "
                         "or call NHS 111 if you are worried.")
        parts.append("**Get help sooner** (GP urgently, NHS 111, or 999 if very unwell) if you develop a high temperature, shivering, "
                     "pain in your back or side, blood in your urine, can't pass urine, or feel confused.")
        return "\n\n".join(parts)

    @staticmethod
    def _out(text, level):
        return {"response": f"{text}\n\n*{PATIENT_DISCLAIMER}*", "level": level}
