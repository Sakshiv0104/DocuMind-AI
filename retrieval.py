"""Vector store and retrieval"""
from langchain_community.vectorstores import FAISS
import numpy as np


class VectorStore:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.store = None
    
    def build(self, documents, embeddings):
        """Build FAISS vector store"""
        embeddings_array = np.array(embeddings).astype('float32')
        
        class EmbedWrapper:
            def __init__(self, fn):
                self.fn = fn
            def embed_documents(self, texts):
                return [self.fn(t) for t in texts]
            def embed_query(self, text):
                return self.fn(text)
        
        self.store = FAISS.from_embeddings(
            text_embeddings=[(doc.page_content, emb) for doc, emb in zip(documents, embeddings_array)],
            embedding=EmbedWrapper(self.embed_fn),
            metadatas=[doc.metadata for doc in documents]
        )
    
    def search(self, query_embedding, k=5):
        """Search for similar documents"""
        return self.store.similarity_search_by_vector(embedding=query_embedding, k=k)
