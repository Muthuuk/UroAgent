"""Hugging Face Spaces / local entry point:  python app.py"""
from uroagent.agent import UroAgent
from uroagent.webapp import build_app

agent = UroAgent(memory_path="uroagent_memory.sqlite")
demo = build_app(agent)

if __name__ == "__main__":
    demo.launch()
