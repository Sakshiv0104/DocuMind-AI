"""Main RAG pipeline with multi-document support"""
from rag.embeddings import CLIPEmbedder
from rag.processing import PDFProcessor
from rag.word_processor import WordProcessor
from rag.retrieval import VectorStore
from rag.generation import Generator
from config import TOP_K
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline"""
        print("Loading models...")
        self.embedder = CLIPEmbedder()
        self.pdf_processor = PDFProcessor()
        self.word_processor = WordProcessor()
        self.vector_store = VectorStore(self.embedder.embed_text)
        self.generator = Generator()
        self.documents = []
        self.all_embeddings = []
        self.combined_image_store = {}
        print("✓ Ready!")
    
    def index_document(self, file_path):
        """
        Index a document (PDF or Word)
        
        Args:
            file_path: Path to PDF or Word file
        """
        file_path = Path(file_path)
        print(f"Processing: {file_path.name}")
        
        # Determine file type and process
        if file_path.suffix.lower() == '.pdf':
            new_docs = self.pdf_processor.process_pdf(file_path)
            self.combined_image_store.update(self.pdf_processor.image_store)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            new_docs = self.word_processor.process_word(file_path)
            self.combined_image_store.update(self.word_processor.image_store)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Create embeddings for new documents
        new_embeddings = []
        for doc in new_docs:
            if doc.metadata['type'] == 'text':
                embedding = self.embedder.embed_text(doc.page_content)
            else:  # image
                embedding = self.embedder.embed_image(doc.metadata['image_obj'])
            
            new_embeddings.append(embedding)
        
        # Add to collection
        self.documents.extend(new_docs)
        self.all_embeddings.extend(new_embeddings)
        
        # Rebuild vector store with all documents
        self.vector_store.build(self.documents, self.all_embeddings)
        
        print(f"✓ Indexed {len(new_docs)} items from {file_path.name}")
        return len(new_docs)
    
    def query(self, question):
        """Query across all indexed documents"""
        if not self.documents:
            return "No documents indexed yet. Please upload documents first."
        
        query_emb = self.embedder.embed_text(question)
        results = self.vector_store.search(query_emb, k=TOP_K)
        
        # Use combined image store
        prompt = self.generator.build_prompt(question, results, self.combined_image_store)
        return self.generator.generate(prompt)
