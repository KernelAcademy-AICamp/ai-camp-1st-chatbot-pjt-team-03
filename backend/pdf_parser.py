import pdfplumber
from typing import List


def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """
    텍스트를 청크로 나누기

    Args:
        text: 전체 텍스트
        chunk_size: 청크당 글자 수
        overlap: 청크 간 중복 글자 수

    Returns:
        [{"text": "청크 내용", "index": 0}, ...]
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():  # 빈 청크 제외
            chunks.append(
                {"text": chunk_text, "index": len(chunks), "start": start, "end": end}
            )

        start = end - overlap

    return chunks
