"""Gradio interface for multimodal RAG system"""
import gradio as gr
from rag.pipeline import RAGPipeline
import os

# Initialize pipeline
pipeline = RAGPipeline()

def upload_documents(files):
    """Handle document uploads"""
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
    """Handle user questions"""
    if not question.strip():
        return history + [("", "⚠️ Please enter a question")]
    
    try:
        answer = pipeline.query(question)
        history.append((question, answer))
        return history
    except Exception as e:
        history.append((question, f"❌ Error: {str(e)}"))
        return history

def clear_all():
    """Clear everything"""
    global pipeline
    pipeline = RAGPipeline()
    return None, [], ""

# Create Gradio interface
with gr.Blocks(
    title="📚 DocuMind AI",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple"
    )
) as demo:
    gr.Markdown("# 📚 DocuMind AI")
    gr.Markdown("### Multimodal RAG System for PDF and Word Documents")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Upload Documents")
            gr.Markdown("Upload up to 5 PDF or Word documents (max 5 pages each)")
            
            file_input = gr.File(
                label="Select Files",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".doc"]
            )
            upload_btn = gr.Button("📥 Upload & Process", variant="primary")
            upload_status = gr.Textbox(
                label="Upload Status",
                lines=5,
                interactive=False
            )
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        with gr.Column(scale=2):
            gr.Markdown("## 💬 Ask Questions")
            gr.Markdown("Ask questions about your uploaded documents")
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400
            )
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask something about your documents...",
                lines=2
            )
            ask_btn = gr.Button("🔍 Ask", variant="primary")
    
    # Event handlers
    upload_btn.click(
        fn=upload_documents,
        inputs=[file_input],
        outputs=[upload_status]
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[question_input]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[question_input]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[file_input, chatbot, upload_status]
    )
    
    gr.Markdown("---")
    gr.Markdown("🤖 **Powered by CLIP, Groq, and LangChain** | Built with ❤️")

# Launch for Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
