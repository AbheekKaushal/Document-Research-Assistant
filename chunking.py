from bisect import bisect_right
import hashlib
import re

from config import DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS, MAX_CHUNKS
from models import Chunk, Page


def chunk_pages(pages: list[Page], size=DEFAULT_CHUNK_WORDS,
                overlap=DEFAULT_OVERLAP_WORDS) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Chunk size must be positive and greater than overlap.")
    chunks = []
    for page in pages:
        words = list(re.finditer(r"\S+", page.text))
        line_breaks = [match.start() for match in re.finditer("\n", page.text)]
        for start in range(0, len(words), size - overlap):
            end = min(start + size, len(words))
            first_char, last_char = words[start].start(), words[end - 1].end()
            text = page.text[first_char:last_char]
            signature = f"{page.document_id}:{page.page_number}:{first_char}:{last_char}:{text}"
            chunk_id = "c_" + hashlib.sha256(signature.encode()).hexdigest()[:20]
            chunks.append(Chunk(
                chunk_id, page.document_id, page.filename, text, page.page_number,
                bisect_right(line_breaks, first_char) + 1,
                bisect_right(line_breaks, last_char - 1) + 1,
            ))
            if len(chunks) > MAX_CHUNKS:
                raise ValueError(f"Too many passages; use fewer documents (limit {MAX_CHUNKS}).")
            if end == len(words):
                break
    return chunks
