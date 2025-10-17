import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class VectorStore:
    """FAISS 기반 벡터 저장소"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str = None):
        print(f"임베딩 모델 로딩 중: {model_name}")

        # 캐시 폴더 지정 (프로젝트 내부에 저장 가능)
        if cache_folder is None:
            cache_folder = "./models"  # 프로젝트 폴더 안에 저장

        Path(cache_folder).mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []

    def create_index(self, chunks: List[dict]):
        """
        청크들로부터 FAISS 인덱스 생성

        Args:
            chunks: [{"text": "...", "index": 0}, ...]
        """
        print(f"임베딩 생성 중... ({len(chunks)}개 청크)")

        # 텍스트만 추출
        texts = [chunk["text"] for chunk in chunks]

        # 임베딩 생성
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # FAISS 인덱스 생성
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        # 청크 메타데이터 저장
        self.chunks = chunks

        print(f"✅ 인덱스 생성 완료: {len(chunks)}개 청크")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """
        쿼리와 유사한 청크 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수

        Returns:
            [(chunk, score), ...] - score가 낮을수록 유사
        """
        if self.index is None:
            return []

        # 쿼리 임베딩
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        # 검색
        distances, indices = self.index.search(query_embedding, top_k)

        # 결과 반환
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))

        return results

    def save(self, path: str):
        """인덱스 저장"""
        Path(path).mkdir(parents=True, exist_ok=True)

        # FAISS 인덱스 저장
        faiss.write_index(self.index, f"{path}/index.faiss")

        # 메타데이터 저장
        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"✅ 인덱스 저장 완료: {path}")

    def load(self, path: str):
        """인덱스 로드"""
        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print(f"✅ 인덱스 로드 완료: {len(self.chunks)}개 청크")
