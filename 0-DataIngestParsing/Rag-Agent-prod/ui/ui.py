# ui/ui.py
import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

def ask(message: str) -> str:
    resp = requests.post(API_URL, json={"message": message})
    return resp.json()["answer"]

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=3, placeholder="Ask me something…"),
    outputs=gr.Textbox(),
    title="RAG‑ReAct Agent",
    description="Powered by Groq LLaMA‑3.3‑70b + local FAISS retriever",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)