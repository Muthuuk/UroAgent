"""Hugging Face Spaces / local entry point:  python app.py

Set UROAGENT_LOAD_VISION=1 (done automatically on the Hugging Face Space) to load the
MedGemma image model; this needs a GPU and accepted MedGemma terms on Hugging Face.
"""
from uroagent.agent import UroAgent
from uroagent.imaging import load_vision_backend
from uroagent.patient import PatientAssistant
from uroagent.webapp import build_app

try:
    import spaces          # present on Hugging Face ZeroGPU hardware
except ImportError:
    spaces = None

vision = load_vision_backend()
if spaces is not None:
    if getattr(vision, "available", False):
        _describe = vision.describe

        @spaces.GPU(duration=30)              # short GPU slot so it fits the small free ZeroGPU allowance
        def _gpu_describe(image, modality, question):
            return _describe(image, modality, question)

        vision.describe = _gpu_describe   # GPU attached only while describing an image
    else:
        @spaces.GPU(duration=5)
        def _zerogpu_placeholder():   # ZeroGPU expects at least one GPU function
            return "ok"

agent = UroAgent(memory_path="uroagent_memory.sqlite")
demo = build_app(agent, PatientAssistant(agent), vision)

if __name__ == "__main__":
    demo.launch()
