"""
AgentClaw v6 — Demo UI 入口（向后兼容包装器）
实际实现在 demo/ui.py
"""
from demo.ui import *

if __name__ == "__main__":
    from demo.ui import build_ui
    import gradio as gr

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        theme=gr.themes.Soft(),
        css="""
            .tab-nav button { font-size: 15px; padding: 10px 20px; }
            .main-title { text-align: center; margin-bottom: 10px; }
        """,
    )
