"""PDF and text processing"""
import fitz
import io
import base64
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP


class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.image_store = {}
    
    def process_pdf(self, pdf_path):
        """Extract text and images from PDF"""
        doc = fitz.open(pdf_path)
        all_docs = []
        
        for page_num, page in enumerate(doc):
            # Process text
            text = page.get_text()
            if text.strip():
                temp_doc = Document(page_content=text, metadata={"page": page_num, "type": "text"})
                chunks = self.text_splitter.split_documents([temp_doc])
                all_docs.extend(chunks)
            
            # Process images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                pil_image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                
                image_id = f"page_{page_num}_img_{img_index}"
                
                # Store as base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                self.image_store[image_id] = base64.b64encode(buffered.getvalue()).decode()
                
                # Add to documents
                all_docs.append(Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": page_num, "type": "image", "image_id": image_id, "image_obj": pil_image}
                ))
        
        doc.close()
        return all_docs
