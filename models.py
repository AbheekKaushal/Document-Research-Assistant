from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Page:
    document_id: str
    filename: str
    text: str
    page_number: Optional[int] = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_number: Optional[int]
    start_line: int
    end_line: int

    @property
    def location(self):
        if self.page_number is not None:
            return f"PDF page {self.page_number}"
        return f"lines {self.start_line}–{self.end_line}"


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    evidence_score: float


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class Claim:
    text: str
    citations: list[Citation]


@dataclass
class Answer:
    question: str
    status: str
    mode: str
    claims: list[Claim] = field(default_factory=list)
    message: str = ""
    model: str = ""
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0
    retrieval_query: str = ""
    retrieval_strategy: str = "search"
    context_questions: list[str] = field(default_factory=list)
    coverage_note: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class IngestionResult:
    pages: list[Page] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalise(text):
    """Ignore whitespace differences when checking exact supporting quotes."""
    return " ".join(text.split())
