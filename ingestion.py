"""Read PDF/TXT/MD bytes without writing user documents to disk."""
import hashlib
from io import BytesIO
from pathlib import Path

from config import MAX_FILE_BYTES, MAX_PDF_PAGES, MAX_TEXT_CHARACTERS
from models import IngestionResult, Page


def read_document(filename: str, content: bytes) -> IngestionResult:
    # Never use an upload's filename as a filesystem path.
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".txt", ".md"}:
        raise ValueError("Supported formats are PDF, TXT, and Markdown.")
    if not content:
        raise ValueError(f"{filename}: the file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError(f"{filename}: the file exceeds the 20 MB limit.")
    document_id = hashlib.sha256(content).hexdigest()[:16]
    result = IngestionResult()

    if extension in {".txt", ".md"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError(f"{filename}: save the text file using UTF-8 encoding.") from error
        if len(text) > MAX_TEXT_CHARACTERS:
            raise ValueError(f"{filename}: extracted text exceeds the size limit.")
        if text.strip():
            result.pages.append(Page(document_id, filename, text))
        else:
            result.warnings.append(f"{filename}: no readable text was found.")
        return result

    from pypdf import PdfReader
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("password-protected PDFs are not supported")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError(f"PDFs must contain at most {MAX_PDF_PAGES} pages")
        extracted_length = 0
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            extracted_length += len(text)
            if extracted_length > MAX_TEXT_CHARACTERS:
                raise ValueError("extracted text exceeds the size limit")
            if text.strip():
                result.pages.append(Page(document_id, filename, text, number))
            else:
                result.warnings.append(
                    f"{filename}, page {number}: no text layer; OCR may be needed."
                )
    except Exception as error:
        raise ValueError(f"{filename}: cannot read PDF ({error}).") from error
    return result

