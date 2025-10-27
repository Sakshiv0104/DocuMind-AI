"""Configuration settings"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Settings
CLIP_MODEL = "openai/clip-vit-base-patch32"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
