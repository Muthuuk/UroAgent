"""Hugging Face Spaces / local entry point:  python app.py"""
from uroagent.agent import UroAgent
from uroagent.webapp import build_app

try:
    # On Hugging Face 'ZeroGPU' hardware the app runs on CPU; ZeroGPU expects at least one
    # GPU-decorated function, so this tiny one is declared. UroAgent itself needs no GPU.
    import spaces

    @spaces.GPU(duration=5)
    def _zerogpu_placeholder():
        return "ok"
except ImportError:
    pass

agent = UroAgent(memory_path="uroagent_memory.sqlite")
demo = build_app(agent)

if __name__ == "__main__":
    demo.launch()
