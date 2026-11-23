"""Typed state shared by every node of the UroAgent graph (case study section 4.3.1)."""
from typing import List, Optional, TypedDict


class UroAgentState(TypedDict, total=False):
    # Input
    messages: List[dict]            # conversation history: {"role": ..., "content": ...}
    clinical_note: str              # raw clinical narrative or question
    patient_id: Optional[str]       # pseudonymised reference only (never a real identifier)
    thread_id: str                  # conversation id used by the memory checkpointer
    clinical_context: Optional[dict]  # earlier findings and assessment in this thread (from memory)
    # Triage
    query_type: str                 # 'diagnosis' | 'scoring' | 'guideline' | 'oob'
    entities: dict                  # extracted clinical entities
    red_flags: List[str]            # safety findings that need urgent action
    # Classification
    prediction: str                 # 'OAB' | 'BPS' | 'UTI' | 'Indeterminate'
    confidence: float
    probabilities: dict
    feature_importances: dict       # SHAP contributions for the predicted class
    # Retrieval
    retrieved_context: List[dict]
    guideline_citations: List[str]
    # Scores
    iciq_oab: Optional[dict]
    oleary_sant: Optional[dict]
    # Output
    top_class: str
    explanation: str
    soap_report: str
    response: str
    escalate: bool
    trace: List[str]
    latency_s: float                # step-by-step reasoning trace shown to the clinician
