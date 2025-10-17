import os
import sys
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import shutil

# 프로젝트 루트를 sys.path에 추가 (이 파일 위치 기준)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# .env 파일 로드
load_dotenv(ROOT_DIR / '.env')

# 환경변수 확인 및 안내
def check_environment():
    """환경변수 설정 상태 확인"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        st.error("⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        st.info("""
        **설정 방법:**
        1. 프로젝트 루트에 `.env` 파일 생성
        2. `.env` 파일에 다음 내용 추가:
           ```
           ANTHROPIC_API_KEY=실제_API_키
           ```
        3. 앱 재시작
        """)
        return False
    return True

from backend.rag import RAGSystem


@st.cache_resource
def get_rag_system(cache_dir: str):
    # 최초 1회만 실제 모델 로드됨 (이후 캐시)
    try:
        return RAGSystem(model_name="all-MiniLM-L6-v2", cache_dir=cache_dir)
    except RuntimeError as e:
        if "ANTHROPIC_API_KEY" in str(e):
            st.error("⚠️ ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
            st.info("환경변수를 설정하거나 .env 파일을 생성해주세요.")
            return None
        else:
            raise e


# 페이지 설정
st.set_page_config(page_title="학습 챗봇", page_icon="📚", layout="wide")

# 업로드 디렉토리 및 사용자 PDF 인덱스 초기화 (접속 시 정리)
if "uploads_initialized" not in st.session_state:
    # 업로드 디렉토리 정리
    UPLOAD_DIR = Path("./data/uploads")
    if UPLOAD_DIR.exists():
        try:
            shutil.rmtree(UPLOAD_DIR)
        except Exception as e:
            st.warning(f"업로드 디렉토리 정리 실패: {e}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # 사용자 PDF 인덱스 디렉토리 정리
    USER_PDF_INDEX_DIR = Path("./data/user_pdf_vectorstore")
    if USER_PDF_INDEX_DIR.exists():
        try:
            shutil.rmtree(USER_PDF_INDEX_DIR)
        except Exception as e:
            st.warning(f"사용자 PDF 인덱스 디렉토리 정리 실패: {e}")
    
    st.session_state.uploads_initialized = True

# 세션 스테이트 초기화
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_loaded" not in st.session_state:
    st.session_state.knowledge_loaded = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader-0"

# 인덱스 파일 존재 여부 확인 (배경지식과 사용자 PDF 분리)
knowledge_index_dir = Path("./data/knowledge_vectorstore")
user_pdf_index_dir = Path("./data/user_pdf_vectorstore")
has_knowledge_index = (knowledge_index_dir / "index.faiss").exists() and (knowledge_index_dir / "chunks.pkl").exists()
has_user_pdf_index = (user_pdf_index_dir / "index.faiss").exists() and (user_pdf_index_dir / "chunks.pkl").exists()
# 레거시 인덱스 관련 로직 제거됨

# 제목
st.title("📚 학습 지원 챗봇")
st.markdown("PDF를 업로드하고 AI에게 질문하세요!")

# 환경변수 확인
if not check_environment():
    st.stop()

# 사이드바 - PDF 업로드
with st.sidebar:
    st.header("📁 문서 업로드")

    uploaded_file = st.file_uploader(
        "PDF 파일 선택", type=["pdf"], key=st.session_state.uploader_key, help="강의자료 PDF를 업로드하세요"
    )

    # 배경지식 필수 로드
    if not has_knowledge_index:
        st.error("❌ 배경지식 인덱스가 없습니다!")
        st.info(f"""
        **필요한 파일:**
        - {knowledge_index_dir}/index.faiss
        - {knowledge_index_dir}/chunks.pkl
        
        **생성 방법:**
        ```bash
        python -m backend.knowledge_indexer --build-index --index-out ./data/knowledge_vectorstore
        ```
        """)
        st.stop()
    
    # 배경지식 인덱스 자동 로드
    if st.session_state.rag_system is None:
        with st.spinner("📚 배경지식 로딩 중..."):
            cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "./models")
            st.session_state.rag_system = get_rag_system(cache_dir)
            if st.session_state.rag_system is not None:
                try:
                    st.session_state.rag_system.load_knowledge_index()
                    st.session_state.knowledge_loaded = True
                    st.success("✅ 배경지식 로드 완료")
                except Exception as e:
                    st.error(f"배경지식 로드 실패: {e}")
                    st.stop()

    if uploaded_file:
        # 파일이 업로드되면 자동으로 인덱싱 시작
        file_path = f"./data/uploads/{uploaded_file.name}"
        
        # 같은 파일이 이미 업로드되어 있는지 확인
        if not os.path.exists(file_path) or os.path.getsize(file_path) != uploaded_file.size:
            # 파일 저장
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 자동 인덱싱 진행
            with st.spinner("📄 파일 업로드 및 인덱싱 중... (1-2분 소요)"):
                cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "./models")
                if st.session_state.rag_system is None:
                    st.session_state.rag_system = get_rag_system(cache_dir)
                
                if st.session_state.rag_system is None:
                    st.error("⚠️ RAG 시스템을 초기화할 수 없습니다. API 키를 확인해주세요.")
                else:
                    try:
                        num_chunks = st.session_state.rag_system.index_user_pdf(file_path)
                        st.session_state.rag_system.save_user_pdf_index()
                        st.session_state.indexed = True
                        st.success(f"✅ 업로드 및 인덱싱 완료! ({num_chunks}개 청크)")
                        st.session_state.uploader_key = f"uploader-{int(time.time())}"
                        st.rerun()
                    except Exception as e:
                        st.error(f"오류: {e}")
        else:
            # 이미 같은 파일이 있는 경우: 사용자 인덱스 존재 여부 확인
            if has_user_pdf_index:
                st.info("📄 이미 업로드된 파일입니다. 인덱싱을 다시 시작하려면 '인덱싱 다시 시작' 버튼을 클릭하세요.")
                if st.button("🔄 인덱싱 다시 시작", type="secondary"):
                    with st.spinner("📄 인덱싱 다시 시작... (1-2분 소요)"):
                        cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "./models")
                        if st.session_state.rag_system is None:
                            st.session_state.rag_system = get_rag_system(cache_dir)
                        
                        if st.session_state.rag_system is None:
                            st.error("⚠️ RAG 시스템을 초기화할 수 없습니다. API 키를 확인해주세요.")
                        else:
                            try:
                                num_chunks = st.session_state.rag_system.index_user_pdf(file_path)
                                st.session_state.rag_system.save_user_pdf_index()
                                st.session_state.indexed = True
                                st.success(f"✅ 인덱싱 완료! ({num_chunks}개 청크)")
                                st.session_state.uploader_key = f"uploader-{int(time.time())}"
                                st.rerun()
                            except Exception as e:
                                st.error(f"오류: {e}")
            else:
                # 파일은 있으나 사용자 인덱스가 없으면 즉시 인덱싱 수행
                with st.spinner("📄 인덱싱 중... (1-2분 소요)"):
                    cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "./models")
                    if st.session_state.rag_system is None:
                        st.session_state.rag_system = get_rag_system(cache_dir)
                    
                    if st.session_state.rag_system is None:
                        st.error("⚠️ RAG 시스템을 초기화할 수 없습니다. API 키를 확인해주세요.")
                    else:
                        try:
                            num_chunks = st.session_state.rag_system.index_user_pdf(file_path)
                            st.session_state.rag_system.save_user_pdf_index()
                            st.session_state.indexed = True
                            st.success(f"✅ 인덱싱 완료! ({num_chunks}개 청크)")
                            st.session_state.uploader_key = f"uploader-{int(time.time())}"
                            st.rerun()
                        except Exception as e:
                            st.error(f"오류: {e}")

    st.divider()

    # 설정
    st.header("⚙️ 설정")
    
    # 상태 표시
    st.success("✅ 배경지식 준비됨")
    
    if st.session_state.indexed:
        st.success("✅ 사용자 문서 준비됨")
    else:
        st.info("📄 PDF를 업로드해주세요")

    top_k = st.slider("검색할 청크 수", 1, 5, 3)

    if st.button("🗑️ 초기화"):
        st.session_state.indexed = False
        st.session_state.chat_history = []
        st.rerun()

# 메인 - 채팅 인터페이스 (배경지식이 필수로 로드됨)
if True:  # 배경지식이 항상 로드되므로 항상 활성화
    st.header("💬 AI 튜터와 대화하기")

    # 채팅 히스토리 표시
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📄 참고 문서 보기"):
                        # 배경지식 소스 표시
                        if "knowledge_sources" in msg and msg["knowledge_sources"]:
                            st.markdown("**📚 배경지식**")
                            for source in msg["knowledge_sources"]:
                                st.markdown(f"**[배경지식 청크 {source['index']}] (유사도: {source['score']})**")
                                st.text(source["text"])
                                st.divider()
                        
                        # PDF 소스 표시
                        if "pdf_sources" in msg and msg["pdf_sources"]:
                            st.markdown("**📄 업로드된 문서**")
                            for source in msg["pdf_sources"]:
                                st.markdown(f"**[문서 청크 {source['index']}] (유사도: {source['score']})**")
                                st.text(source["text"])
                                st.divider()
                        
                        # 기존 소스 구조 호환성 (혹시 모를 경우)
                        if not ("knowledge_sources" in msg or "pdf_sources" in msg):
                            for i, source in enumerate(msg["sources"], 1):
                                st.markdown(f"**[청크 {source['index']}] (유사도: {source['score']})**")
                                st.text(source["text"])
                                st.divider()

    # 입력창
    user_input = st.chat_input("질문을 입력하세요...")

    if user_input:
        # RAG 시스템이 없으면 초기화
        if st.session_state.rag_system is None:
            cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "./models")
            st.session_state.rag_system = get_rag_system(cache_dir)
            
            if st.session_state.rag_system is None:
                st.error("⚠️ RAG 시스템을 초기화할 수 없습니다. API 키를 확인해주세요.")
                st.stop()

        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # RAG 검색 및 답변 생성 (PDF 업로드 여부에 따라 다른 방식 사용)
        with st.spinner("답변 생성 중..."):
            if st.session_state.indexed:
                # PDF가 업로드된 경우: 배경지식 + 사용자 PDF 모두 활용
                result = st.session_state.rag_system.qna_with_knowledge(user_input, top_k=top_k)
            else:
                # PDF가 없는 경우: 배경지식만 활용
                result = st.session_state.rag_system.qna_with_knowledge_only(user_input, top_k=top_k)

        # AI 응답 추가 (배경지식과 PDF 소스를 통합)
        all_sources = result.get("knowledge_sources", []) + result.get("pdf_sources", [])
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": all_sources,
                "knowledge_sources": result.get("knowledge_sources", []),
                "pdf_sources": result.get("pdf_sources", [])
            }
        )

        st.rerun()

else:
    # 이 코드는 실행되지 않음 (배경지식이 필수이므로)
    pass

    st.markdown(
        """
    ### 🎯 사용 방법
    
    1. **배경지식 설정**: JSON 파일로 배경지식을 미리 설정 (선택사항)
    2. **PDF 업로드**: 왼쪽 사이드바에서 강의자료 PDF 선택 (자동 인덱싱)
    3. **질문하기**: 배경지식과 업로드된 문서를 종합하여 질문
    
    ### ✨ RAG 기능
    
    - 📄 문서에서 관련 부분만 찾아서 답변
    - 🎯 정확한 출처 표시
    - ⚡ 긴 문서도 빠르게 처리
    - 💰 토큰 비용 절감
    
    ### 🔍 작동 원리
    
    ```
    질문 입력
      ↓
    FAISS 벡터 검색 (관련 청크 3-5개 찾기)
      ↓
    찾은 내용을 LLM에게 전달
      ↓
    정확한 답변 생성
    ```
    """
    )

# 푸터
st.divider()
