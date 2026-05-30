import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

class Config:
    GROQ_API_KEY: str    = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str       = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    CHROMA_DB_PATH: str         = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "sc_rag_docs")

    TOP_K: int         = int(os.getenv("TOP_K", 5))
    CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", 256))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 32))

    QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", 3.0))
    MAX_RETRIES: int         = int(os.getenv("MAX_RETRIES", 2))

    def validate(self):
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is missing!\n"
                "1. Copy .env.example to .env\n"
                "2. Add your key from https://console.groq.com"
            )
        return self

config = Config()