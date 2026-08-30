"""Root Gradio runner for backwards compatibility."""
from apps.gradio_ui.app import create_ui

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
