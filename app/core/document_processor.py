import re
from pathlib import Path

from app.utils.logger import logger


def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def process_pdf(file_path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        full_text = "\n".join(pages)
        logger.info(f"pdf processed: {len(reader.pages)} pages, {len(full_text)} chars")
        return _clean_text(full_text)
    except Exception as e:
        logger.error(f"pdf processing failed: {e}")
        raise


def process_word(file_path: str) -> str:
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        logger.info(f"word file processed: {len(paragraphs)} paragraphs, {len(full_text)} chars")
        return _clean_text(full_text)
    except Exception as e:
        logger.error(f"word processing failed: {e}")
        raise


def process_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        logger.info(f"txt processed: {len(text)} chars")
        return _clean_text(text)
    except Exception as e:
        logger.error(f"txt processing failed: {e}")
        raise


def process_file(file_path: str) -> dict:
    path = Path(file_path)
    extension = path.suffix.lower()
    filename = path.name

    logger.info(f"processing file: {filename} ({extension})")

    if extension == ".pdf":
        text = process_pdf(file_path)
        file_type = "pdf"
    elif extension in [".docx", ".doc"]:
        text = process_word(file_path)
        file_type = "word"
    elif extension == ".txt":
        text = process_txt(file_path)
        file_type = "txt"
    else:
        raise ValueError(
            f"unsupported file type: {extension}. supported: PDF, Word (.docx), TXT"
        )

    if not text.strip():
        raise ValueError(f"no text could be extracted from {filename}")

    return {
        "text": text,
        "filename": filename,
        "type": file_type,
        "chars": len(text),
    }