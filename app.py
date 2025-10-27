"""
DocuMind AI - Clean Professional Interface
"""

import gradio as gr
from pathlib import Path
from rag.pipeline import RAGPipeline

pipeline = None
processed_files = []

def process_documents(doc_files):
    global pipeline, processed_files
    if not doc_files:
        return "⚠️ Please select documents"
    if len(doc_files) > 5:
        return "⚠️ Maximum 5 documents"
    try:
        if pipeline is None:
            pipeline = RAGPipeline()
        processed_files = []
        status = "Processing...\n\n"
        for doc_file in doc_files:
            file_path = Path(doc_file.name)
            if file_path.suffix.lower() not in ['.pdf', '.docx', '.doc']:
                status += f"❌ {file_path.name} - Unsupported\n"
                continue
            num = pipeline.index_document(file_path)
            processed_files.append(file_path.name)
            status += f"✅ {file_path.name} ({num} items)\n"
        return status + f"\n✅ {len(processed_files)} document(s) ready!"
    except Exception as e:
        return f"❌ {str(e)}"

def answer_question(question, history):
    global pipeline, processed_files
    if not pipeline or not processed_files:
        return history + [[question, "⚠️ Upload documents first"]]
    if not question.strip():
        return history + [[question, "⚠️ Enter a question"]]
    try:
        answer = pipeline.query(question)
        return history + [[question, answer + f"\n\n📚 {', '.join(processed_files)}"]]
    except Exception as e:
        return history + [[question, f"❌ {str(e)}"]]

def reset():
    global pipeline, processed_files
    pipeline = None
    processed_files = []
    return "Reset complete", []

# Use Base theme with custom settings
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="gray",
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_900",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    input_background_fill="white",
    button_primary_background_fill="*primary_600",
    button_primary_text_color="white",
)

with gr.Blocks(theme=theme, title="DocuMind AI") as demo:
    
    gr.HTML("""
    <div style='background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%); padding: 2.5rem; text-align: center; border-radius: 12px 12px 0 0;'>
        <h1 style='color: white; margin: 0; font-size: 2rem;'>🧠 DocuMind AI</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>Intelligent Document Analysis Platform</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Documents")
            doc_input = gr.File(label="Select files (max 5)", file_types=[".pdf", ".docx", ".doc"], file_count="multiple", type="filepath")
            process_btn = gr.Button("🚀 Analyze Documents", variant="primary", size="lg")
            status = gr.Textbox(label="Status", lines=10, interactive=False, placeholder="Ready...")
            reset_btn = gr.Button("🔄 Reset", variant="secondary")
            gr.Markdown("""
            ---
            **How to Use:**
            1. Upload PDF/Word files
            2. Click Analyze
            3. Ask questions
            4. Get AI answers
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(height=450, avatar_images=("👤", "🤖"))
            question = gr.Textbox(label="Your Question", placeholder="Ask anything...", lines=2)
            with gr.Row():
                send = gr.Button("📨 Send", variant="primary", scale=3)
                clear = gr.Button("Clear", variant="secondary", scale=1)
    
    gr.Examples(
        [["What are the main topics?"], ["Summarize the documents"], ["What images are shown?"]],
        question
    )
    
    gr.Markdown("---\n**DocuMind AI** | Powered by Gemini AI | 🔒 Secure Processing")
    
    process_btn.click(process_documents, doc_input, status)
    send.click(answer_question, [question, chatbot], chatbot).then(lambda: "", outputs=question)
    question.submit(answer_question, [question, chatbot], chatbot).then(lambda: "", outputs=question)
    clear.click(lambda: [], outputs=chatbot)
    reset_btn.click(reset, outputs=[status, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
