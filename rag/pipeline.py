"""Main RAG pipeline with lazy-loaded CLIP"""
from rag.embeddings import CLIPEmbedder
from rag.processing import PDFProcessor
from rag.word_processor import WordProcessor
from rag.retrieval import VectorStore
from rag.generation import Generator
from config import TOP_K
from pathlib import Path

class RAGPipeline:
    def __init__(self):
        """Initialize pipeline - CLIP loads on first upload"""
        print("✓ Pipeline ready!")
        self.embedder = None  # Lazy load
        self.pdf_processor = PDFProcessor()
        self.word_processor = WordProcessor()
        self.vector_store = None
        self.generator = Generator()
        self.documents = []
        self.all_embeddings = []
        self.combined_image_store = {}
    
    def _load_clip(self):
        """Load CLIP model on first use"""
        if self.embedder is None:
            print("📥 Loading CLIP model (this may take a few minutes)...")
            self.embedder = CLIPEmbedder()
            self.vector_store = VectorStore(self.embedder.embed_text)
            print("✓ CLIP model ready!")
    
    def index_document(self, file_path):
        """Index a document (PDF or Word)"""
        # Load CLIP if not already loaded
        self._load_clip()
        
        file_path = Path(file_path)
        print(f"📄 Processing: {file_path.name}")
        
        # Process based on file type
        if file_path.suffix.lower() == '.pdf':
            new_docs = self.pdf_processor.process_pdf(file_path)
            self.combined_image_store.update(self.pdf_processor.image_store)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            new_docs = self.word_processor.process_word(file_path)
            self.combined_image_store.update(self.word_processor.image_store)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Create embeddings
        new_embeddings = []
        for doc in new_docs:
            if doc.metadata['type'] == 'text':
                emb = self.embedder.embed_text(doc.page_content)
            else:
                emb = self.embedder.embed_image(doc.metadata['image_obj'])
            new_embeddings.append(emb)
        
        # Store everything
        self.documents.extend(new_docs)
        self.all_embeddings.extend(new_embeddings)
        self.vector_store.build(self.documents, self.all_embeddings)
        
        print(f"✅ Successfully indexed {len(new_docs)} items from {file_path.name}")
        return len(new_docs)
    
    def query(self, question):
        """Query across all indexed documents"""
        if not self.documents:
            return "⚠️ Please upload documents first."
        
        query_emb = self.embedder.embed_text(question)
        results = self.vector_store.search(query_emb, k=TOP_K)
        prompt = self.generator.build_prompt(question, results, self.combined_image_store)
        return self.generator.generate(prompt)
