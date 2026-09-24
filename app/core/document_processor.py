import re
from pathlib import Path

from app.utils.logger import logger


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def process_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages  = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        full_text = "\n".join(pages)
        logger.info(f"PDF processed: {len(reader.pages)} pages, {len(full_text)} chars")
        return _clean_text(full_text)
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise


def process_word(file_path: str) -> str:
    """Extract text from a Word (.docx) file."""
    try:
        from docx import Document
        doc        = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text  = "\n".join(paragraphs)
        logger.info(f"Word file processed: {len(paragraphs)} paragraphs, {len(full_text)} chars")
        return _clean_text(full_text)
    except Exception as e:
        logger.error(f"Word processing failed: {e}")
        raise


def process_txt(file_path: str) -> str:
    """Read plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        logger.info(f"TXT file processed: {len(text)} chars")
        return _clean_text(text)
    except Exception as e:
        logger.error(f"TXT processing failed: {e}")
        raise


def process_file(file_path: str) -> dict:
    """
    Main entry point. Detects file type and extracts text.

    Returns:
        {
            "text":     extracted plain text,
            "filename": original filename,
            "type":     pdf / word / txt,
            "chars":    character count,
        }
    """
    path      = Path(file_path)
    extension = path.suffix.lower()
    filename  = path.name

    logger.info(f"Processing file: {filename} ({extension})")

    if extension == ".pdf":
        text      = process_pdf(file_path)
        file_type = "pdf"
    elif extension in [".docx", ".doc"]:
        text      = process_word(file_path)
        file_type = "word"
    elif extension == ".txt":
        text      = process_txt(file_path)
        file_type = "txt"
    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: PDF, Word (.docx), TXT"
        )

    if not text.strip():
        raise ValueError(f"No text could be extracted from {filename}")

    return {
        "text":     text,
        "filename": filename,
        "type":     file_type,
        "chars":    len(text),
    }