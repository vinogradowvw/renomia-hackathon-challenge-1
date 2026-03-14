import os

from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / '.env')

class Config:
    """Application configuration."""
    
    # GEMINI Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    MODEL = os.getenv('MODEL', 'gemini-pro')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://hackathon:hackathon@db:5432/hackathon')

config = Config()
