!pip install langchain_community
!pip install pypdf
!pip install unstructured
!pip install langchain_openai
!pip install scikit-learn
!pip install faiss-cpu

import os  
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document  # 문서 객체 타입 정의용 (직접 생성 시 사용)
from langchain_openai import ChatOpenAI  # OpenAI GPT 모델을 사용하기 위한 LangChain 클래스
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 구성용 템플릿 유틸
from langchain_openai import OpenAIEmbeddings  # OpenAI의 텍스트 임베딩 모델 (예: text-embedding-3)
from langchain_community.vectorstores import FAISS  # FAISS 벡터 저장소 사용 (로컬 유사도 검색에 적합)
from langchain_core.output_parsers import StrOutputParser  # LLM 출력 결과에서 문자열만 추출
from langchain_core.runnables import RunnablePassthrough  # 입력을 그대로 전달하는 체인 구성용 유틸
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv  # .env 또는 텍스트에서 환경 변수 불러오기

load_dotenv('env.txt')

# 문제생성 테스트용 파일 업로드
loader = PyPDFLoader('document\토마토 재배가이드_8월.pdf')
pages = loader.load()

# 청킹
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

# 벡터 저장소 생성
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embedding)

# 리트리버 생성
retriever = vectorstore.as_retriever(search_kwargs={'k':3})

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_template("""
[참고 자료]
{context}

[사용자 질문]
{input}

[지시 사항]
1. 너는 친절한 시험 문제 도우미 봇이야.
2. 아래 사용자 질문에 대해 참고자료를 기반으로 대답해 줘.
3. 객관식 또는 주관식 중 선택된 형식으로 문제를 만들어.
4. 각각의 문제에 대해 "문제 / 정답 / 해설"을 포함시켜 줘.
5. 문제 수는 최소 3개 이상 5개 미만 포함해 줘.
6. 사용자 질문이 토마토 재배와 무관할 경우 "대답할 수 없습니다"라고 답변해.

[출력 포맷]
문제 유형: 객관식 또는 주관식
문제 1:
문제:
정답:
해설:

...
""")

#체인구성
parser = StrOutputParser()
chain = (
    {'context': retriever, 'input': RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# 사용자 쿼리 하드코딩 - streamlit기반 개발 필요
default_query  = "토마토 재배 적정 온도에 대해 주관식으로 문제를 만들어줘?"
print(chain.invoke(default_query ))
