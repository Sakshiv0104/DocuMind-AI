"""CLIP embeddings for text and images"""
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from config import CLIP_MODEL


class CLIPEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model.eval()
    
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
    def embed_image(self, image) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
