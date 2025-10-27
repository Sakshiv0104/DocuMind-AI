"""Gradio interface for multimodal RAG system"""
import gradio as gr
from rag.pipeline import RAGPipeline
import os

pipeline = RAGPipeline()

def upload_documents(files):
    if not files:
        return "⚠️ Please select files to upload"
    
    results = []
    for file in files:
        try:
            num_indexed = pipeline.index_document(file.name)
            results.append(f"✅ {os.path.basename(file.name)}: {num_indexed} items indexed")
        except Exception as e:
            results.append(f"❌ {os.path.basename(file.name)}: Error - {str(e)}")
    
    return "\n".join(results)

def answer_question(question, history):
    if not question.strip():
        return history
    
    try:
        answer = pipeline.query(question)
        history.append((question, answer))
        return history
    except Exception as e:
        history.append((question, f"❌ Error: {str(e)}"))
        return history

def clear_all():
    global pipeline
    pipeline = RAGPipeline()
    return None, [], ""

with gr.Blocks(title="DocuMind AI") as demo:
    gr.Markdown("# 📚 DocuMind AI")
    gr.Markdown("### Multimodal RAG System for PDF and Word Documents")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📤 Upload Documents")
            file_input = gr.Files(label="Select Files", file_types=[".pdf", ".docx", ".doc"])
            upload_btn = gr.Button("Upload & Process")
            upload_status = gr.Textbox(label="Status", lines=5)
            clear_btn = gr.Button("Clear All")
        
        with gr.Column():
            gr.Markdown("## 💬 Ask Questions")
            chatbot = gr.Chatbot(label="Conversation", height=400)
            question_input = gr.Textbox(label="Your Question", lines=2)
            ask_btn = gr.Button("Ask")
    
    upload_btn.click(fn=upload_documents, inputs=file_input, outputs=upload_status)
    ask_btn.click(fn=answer_question, inputs=[question_input, chatbot], outputs=chatbot)
    question_input.submit(fn=answer_question, inputs=[question_input, chatbot], outputs=chatbot)
    clear_btn.click(fn=clear_all, outputs=[file_input, chatbot, upload_status])
    
    gr.Markdown("🤖 Powered by CLIP, Groq, and LangChain")

if __name__ == "__main__":
    demo.queue().launch()
