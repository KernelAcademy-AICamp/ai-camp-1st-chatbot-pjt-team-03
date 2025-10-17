import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Optional, Union
from backend.vectorstore import VectorStore

# 실행방법
# python -m backend.knowledge_indexer --in-dir ./data/knowledge --out ./data/knowledge/merged.txt --build-index --index-out ./data/knowledge_vectorstore

def _extract_ko_txt(payload: Union[dict, list]) -> str:
    """
    JSON payload에서 corpus -> ko_info -> ko_txt 값들만 수집해 하나의 문자열로 병합.
    파일 하나당 반환 문자열 1개.
    """
    parts: List[str] = []

    def add(value: Optional[str]):
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    def walk_ko_info(ko_info: Union[list, dict]):
        if isinstance(ko_info, list):
            for item in ko_info:
                if isinstance(item, dict):
                    add(item.get("ko_txt"))
        elif isinstance(ko_info, dict):
            add(ko_info.get("ko_txt"))

    def walk_corpus(corpus: Union[list, dict]):
        if isinstance(corpus, list):
            for entry in corpus:
                if isinstance(entry, dict) and "ko_info" in entry:
                    walk_ko_info(entry.get("ko_info"))
        elif isinstance(corpus, dict):
            if "ko_info" in corpus:
                walk_ko_info(corpus.get("ko_info"))

    if isinstance(payload, dict) and "corpus" in payload:
        walk_corpus(payload.get("corpus"))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "corpus" in item:
                walk_corpus(item.get("corpus"))

    return "\n".join(parts)


def read_all_json_files(input_dir: str) -> List[str]:
    contents: List[str] = []
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {input_dir}")

    for fp in json_files:
        try:
            raw = fp.read_text(encoding="utf-8")
            if not raw.strip():
                continue
            payload = json.loads(raw)
            merged_ko = _extract_ko_txt(payload)
            if merged_ko.strip():
                contents.append(merged_ko)
        except Exception as e:
            print(f"⚠️ 파일 처리 실패: {fp} -> {e}")

    return contents


def write_merged_file(docs: List[str], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 파일 간 구분은 두 줄바꿈, 파일 내부 ko_txt 병합은 한 줄바꿈
    merged = "\n\n".join(docs)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(merged)
    print(f"✅ 병합 완료: {len(docs)}개 파일 -> {output_file}")


def build_index_from_merged(
    merged_path: str = "./data/knowledge/merged.txt",
    output_dir: str = "./data/knowledge_vectorstore",
    model_name: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    cache_dir: Optional[str] = os.environ.get("EMBEDDING_CACHE_DIR", "./models"),
):
    """
    merged.txt를 두 줄바꿈(연속 개수 무관) 기준으로 청크로 나누어 벡터 인덱스를 생성합니다.
    """
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"병합 파일을 찾을 수 없습니다: {merged_path}")

    with open(merged_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 두 줄 이상의 줄바꿈을 구분자로 사용하여 분리
    parts = [seg.strip() for seg in re.split(r"\n{2,}", text) if seg.strip()]
    chunks = [{"text": seg, "index": i} for i, seg in enumerate(parts)]

    if not chunks:
        raise ValueError("생성된 청크가 없습니다. merged.txt 내용을 확인하세요.")

    store = VectorStore(model_name=model_name, cache_folder=cache_dir)
    store.create_index(chunks)

    os.makedirs(output_dir, exist_ok=True)
    store.save(output_dir)
    print(f"✅ 인덱스 생성 완료: {len(chunks)}개 청크 -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="data/knowledge의 모든 JSON 병합 및(옵션) merged.txt 인덱싱")
    parser.add_argument("--in-dir", default="./data/knowledge", help="입력 디렉토리 (기본: ./data/knowledge)")
    parser.add_argument("--out", default="./data/knowledge/merged.txt", help="출력 파일 경로 (기본: ./data/knowledge/merged.txt)")
    parser.add_argument("--build-index", action="store_true", help="merged.txt를 두 줄바꿈 기준으로 청크화하여 벡터 인덱스 생성")
    parser.add_argument("--index-out", default="./data/knowledge_vectorstore", help="인덱스 저장 경로 (기본: ./data/knowledge_vectorstore)")
    parser.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), help="SentenceTransformer 모델명")
    parser.add_argument("--cache", default=os.environ.get("EMBEDDING_CACHE_DIR", "./models"), help="임베딩 모델 캐시 디렉토리")

    args = parser.parse_args()

    in_dir = args["in_dir"] if isinstance(args, dict) else args.in_dir
    out_file = args["out"] if isinstance(args, dict) else args.out

    docs = read_all_json_files(in_dir)
    write_merged_file(docs, out_file)

    if args["build_index"] if isinstance(args, dict) else args.build_index:
        index_out = args["index_out"] if isinstance(args, dict) else args.index_out
        model_name = args["model"] if isinstance(args, dict) else args.model
        cache_dir = args["cache"] if isinstance(args, dict) else args.cache
        build_index_from_merged(out_file, output_dir=index_out, model_name=model_name, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
