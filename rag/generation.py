"""LLM generation with Google Gemini"""
import google.generativeai as genai
from config import GEMINI_API_KEY
import base64
from io import BytesIO
from PIL import Image


class Generator:
    def __init__(self):
        """Initialize Gemini model"""
        genai.configure(api_key=GEMINI_API_KEY)
        # Use Gemini 2.0 Flash - free and multimodal
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def build_prompt(self, query, docs, image_store):
        """Build multimodal prompt for Gemini"""
        # Gemini accepts a list of content parts
        parts = []
        
        # Add query
        parts.append(f"Question: {query}\n\nContext:\n")
        
        # Separate text and images
        text_docs = [d for d in docs if d.metadata.get("type") == "text"]
        image_docs = [d for d in docs if d.metadata.get("type") == "image"]
        
        # Add text context
        if text_docs:
            text_context = "\n\n".join([
                f"[Page {d.metadata['page']+1}]: {d.page_content}"
                for d in text_docs
            ])
            parts.append(f"Text excerpts:\n{text_context}\n")
        
        # Add images - Gemini can accept PIL Images directly
        if image_docs:
            parts.append("\nImages from the document:\n")
            for doc in image_docs:
                img_id = doc.metadata.get("image_id")
                if img_id and img_id in image_store:
                    # Decode base64 to PIL Image
                    img_base64 = image_store[img_id]
                    img_bytes = base64.b64decode(img_base64)
                    pil_image = Image.open(BytesIO(img_bytes))
                    parts.append(pil_image)
                    parts.append(f"[Image from page {doc.metadata['page']+1}]\n")
        
        parts.append("\nPlease answer the question based on the provided text and images.")
        
        return parts
    
    def generate(self, parts):
        """Generate response using Gemini"""
        response = self.model.generate_content(parts)
        return response.text
