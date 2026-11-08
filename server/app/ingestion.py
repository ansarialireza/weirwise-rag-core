import csv
import io
import os
import re
from typing import List
from pypdf import PdfReader


def extract_text_from_file(file_path: str, mime_type: str) -> str:
    """Extract clean text from PDF, DOCX, CSV, Markdown, or plain text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    lower_path = file_path.lower()
    is_pdf = "pdf" in (mime_type or "").lower() or lower_path.endswith(".pdf")
    is_docx = "wordprocessingml" in (mime_type or "").lower() or lower_path.endswith(".docx")
    is_csv = "csv" in (mime_type or "").lower() or lower_path.endswith(".csv")

    if is_pdf:
        with open(file_path, "rb") as fh:
            if not fh.read(5).startswith(b"%PDF-"):
                raise ValueError("File is not a valid PDF (missing %PDF- signature)")
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)

    if is_docx:
        import docx

        document = docx.Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))
        return "\n\n".join(paragraphs)

    if is_csv:
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return ""
            header = rows[0]
            lines = []
            for row in rows[1:]:
                if not any(row):
                    continue
                line_parts = [f"{h.strip()}: {v.strip()}" for h, v in zip(header, row) if v.strip()]
                lines.append(" | ".join(line_parts))
            return "\n".join(lines)

    # Plain text, markdown, etc. Reject undecodable bytes instead of indexing replacement chars.
    with open(file_path, "r", encoding="utf-8", errors="strict") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Splits text into overlapping chunks respecting paragraph or sentence boundaries."""
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)
        # If adding this paragraph fits within chunk_size:
        added_len = para_len if not current_chunk else para_len + 2
        if current_length + added_len <= chunk_size:
            current_chunk.append(para)
            current_length += added_len
        else:
            if current_chunk:
                chunk_str = "\n\n".join(current_chunk)
                chunks.append(chunk_str)
                # Overlap handling
                overlap_text = chunk_str[-chunk_overlap:] if len(chunk_str) > chunk_overlap else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_length = sum(len(p) for p in current_chunk) + (2 if overlap_text else 0)
            else:
                # Single paragraph exceeds chunk_size, split by sentences or characters
                for i in range(0, para_len, chunk_size - chunk_overlap):
                    sub = para[i : i + chunk_size]
                    if sub.strip():
                        chunks.append(sub.strip())
                current_chunk = []
                current_length = 0

    if current_chunk:
        chunk_str = "\n\n".join(current_chunk).strip()
        if chunk_str:
            chunks.append(chunk_str)

    return chunks
