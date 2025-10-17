import os
from dotenv import load_dotenv

from anthropic import Anthropic

# .env 파일 로드
load_dotenv()
from backend.vectorstore import VectorStore
from backend.pdf_parser import extract_text_from_pdf, chunk_text
from backend.summarize import DocumentSummarizer
from backend.question_generation import QuestionGenerator
from backend.qna import QnASystem
from typing import List, Tuple


class RAGSystem:
    """RAG 시스템 - 배경지식과 사용자 PDF를 분리하여 처리"""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str | None = None,
        knowledge_index_path: str | None = None
    ):
        # 임베딩 캐시 폴더 결정 (env > 인자 > 기본)
        if cache_dir is None:
            cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "./models")

        # 배경지식용 벡터스토어 (영구 저장)
        self.knowledge_vectorstore = VectorStore(model_name, cache_folder=cache_dir)
        
        # 사용자 PDF용 벡터스토어 (임시 저장)
        self.user_pdf_vectorstore = VectorStore(model_name, cache_folder=cache_dir)

        # 사전 생성된 배경지식 인덱스 자동 로드 (옵션)
        knowledge_index_path = (
            knowledge_index_path
            or os.environ.get("KNOWLEDGE_INDEX_PATH")
            or "./data/knowledge_vectorstore"
        )
        if knowledge_index_path and os.path.exists(knowledge_index_path):
            try:
                self.knowledge_vectorstore.load(knowledge_index_path)
                print(f"🔄 배경지식 인덱스 자동 로드: {knowledge_index_path}")
            except Exception as e:
                print(f"⚠️ 배경지식 인덱스 로드 실패: {e}")

        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        self.client = None
        self.document_name = None
        
        # 전문 모듈들 초기화
        self.summarizer = DocumentSummarizer(self.api_key)
        self.question_generator = QuestionGenerator(self.api_key)
        self.qna_system = QnASystem(self.api_key)

    # 배경지식 인덱스 생성 기능은 모듈(backend/knowledge_indexer.py)로 분리되었습니다.

    def index_user_pdf(self, pdf_path: str, chunk_size: int = 500):
        """
        사용자 PDF 인덱싱 (임시 저장)

        Args:
            pdf_path: PDF 파일 경로
            chunk_size: 청크 크기
        """
        # 파일 존재 여부 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        # 파일 확장자 확인
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"PDF 파일이 아닙니다: {pdf_path}")

        print(f"📄 사용자 PDF 처리 중: {pdf_path}")

        # 1. 텍스트 추출
        text = extract_text_from_pdf(pdf_path)
        print(f"✅ 텍스트 추출 완료: {len(text)} 글자")

        # 2. 청킹
        chunks = chunk_text(text, chunk_size=chunk_size)
        print(f"✅ 청킹 완료: {len(chunks)}개 청크")

        # 3. 사용자 PDF 인덱싱
        self.user_pdf_vectorstore.create_index(chunks)

        # 문서 이름 저장
        self.document_name = os.path.basename(pdf_path)

        return len(chunks)

    def upload_and_index_pdf(self, uploaded_file_content: bytes, filename: str, upload_dir: str = "./data/uploads", chunk_size: int = 500):
        """
        업로드된 파일을 저장하고 인덱싱
        
        Args:
            uploaded_file_content: 업로드된 파일의 바이트 내용
            filename: 파일명
            upload_dir: 업로드 디렉토리
            chunk_size: 청크 크기
            
        Returns:
            인덱싱된 청크 수
        """
        # 업로드 디렉토리 생성
        os.makedirs(upload_dir, exist_ok=True)
        
        # 파일 경로 생성
        file_path = os.path.join(upload_dir, filename)
        
        # 파일 저장
        with open(file_path, "wb") as f:
            f.write(uploaded_file_content)
        
        print(f"📁 파일 저장 완료: {file_path}")
        
        # 인덱싱
        return self.index_user_pdf(file_path, chunk_size)


    def summarize_document(self, top_k: int = 5) -> dict:
        """
        사용자 PDF 문서 요약 (사용자 PDF만 활용)

        Args:
            top_k: 검색할 청크 개수

        Returns:
            {"summary": "요약", "sources": [...]}
        """
        return self.summarizer.summarize(self.user_pdf_vectorstore, top_k)

    def generate_questions(self, num_questions: int = 5) -> dict:
        """
        사용자 PDF 기반 질문 생성 (사용자 PDF만 활용)

        Args:
            num_questions: 생성할 질문 개수

        Returns:
            {"questions": ["질문1", "질문2", ...], "sources": [...]}
        """
        return self.question_generator.generate_questions(self.user_pdf_vectorstore, num_questions)

    def qna_with_knowledge(self, question: str, top_k: int = 3) -> dict:
        """
        Q&A (배경지식 + 사용자 PDF 모두 활용)

        Args:
            question: 사용자 질문
            top_k: 각 인덱스에서 검색할 청크 개수

        Returns:
            {"answer": "답변", "knowledge_sources": [...], "pdf_sources": [...]}
        """
        return self.qna_system.answer_with_knowledge(
            question, 
            self.knowledge_vectorstore, 
            self.user_pdf_vectorstore, 
            top_k
        )

    def generate_questions_by_topic(self, topic: str, num_questions: int = 3) -> dict:
        """
        특정 주제에 대한 질문 생성

        Args:
            topic: 질문을 생성할 주제
            num_questions: 생성할 질문 개수

        Returns:
            {"questions": ["질문1", "질문2", ...], "sources": [...]}
        """
        return self.question_generator.generate_questions_by_topic(
            self.user_pdf_vectorstore, 
            topic, 
            num_questions
        )

    def qna_with_knowledge_only(self, question: str, top_k: int = 3) -> dict:
        """
        배경지식만으로 Q&A

        Args:
            question: 사용자 질문
            top_k: 검색할 청크 개수

        Returns:
            {"answer": "답변", "sources": [...]} 
        """
        return self.qna_system.answer_with_knowledge_only(
            question, 
            self.knowledge_vectorstore, 
            top_k
        )

    def qna_with_pdf_only(self, question: str, top_k: int = 3) -> dict:
        """
        사용자 PDF만으로 Q&A

        Args:
            question: 사용자 질문
            top_k: 검색할 청크 개수

        Returns:
            {"answer": "답변", "sources": [...]} 
        """
        return self.qna_system.answer_with_pdf_only(
            question, 
            self.user_pdf_vectorstore, 
            top_k
        )

    def save_knowledge_index(self, path: str = "./data/knowledge_vectorstore"):
        """배경지식 인덱스 저장"""
        self.knowledge_vectorstore.save(path)

    def load_knowledge_index(self, path: str = "./data/knowledge_vectorstore"):
        """배경지식 인덱스 로드"""
        self.knowledge_vectorstore.load(path)

    def save_user_pdf_index(self, path: str = "./data/user_pdf_vectorstore"):
        """사용자 PDF 인덱스 저장"""
        self.user_pdf_vectorstore.save(path)

    def load_user_pdf_index(self, path: str = "./data/user_pdf_vectorstore"):
        """사용자 PDF 인덱스 로드"""
        self.user_pdf_vectorstore.load(path)

    # 하위 호환성을 위한 기존 메서드들 (deprecated)
    # 레거시 통합 인덱스 관련 메서드는 제거되었습니다.
