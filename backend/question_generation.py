import os
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 세션 상태 저장용 글로벌 캐시 (간단한 메모리)
session_memory = {
    "last_question": None,
    "last_difficulty": "보통",
    "last_file_path": None
}

def generate_questions(user_query: str, question_type: str, pdf_bytes: BytesIO) -> str:

    global session_memory
    try:
        # === 1️⃣ 난이도 인식 및 질의 재구성 ===
        difficulty = "보통"
        if "어렵" in user_query:
            difficulty = "어려움"
        elif "쉽" in user_query or "낮춰" in user_query:
            difficulty = "쉬움"

        # 이전 질의와 결합 (후속 요청인 경우)
        if session_memory["last_question"] and (
            "어렵" in user_query or "쉬움" in user_query or "낮춰" in user_query or "높" in user_query
        ):
            user_query = f"이전 질문 '{session_memory['last_question']}'과 동일한 주제로, 난이도를 {difficulty} 수준으로 조정해줘."

        # === 2️⃣ 파일 캐싱 및 재활용 ===
        pdf_bytes.seek(0)
        temp_path = session_memory["last_file_path"] or "temp_memory_file.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes.read())

        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template("""
        [참고 자료]
        {context}

        [사용자 질문]
        {input}

        [지시 사항]
        1. 너는 친절한 시험 문제 도우미 봇이야.
        2. 아래 사용자 질문에 대해 참고자료를 기반으로 대답해 줘.
        3. 문제는 반드시 "{question_type}" 형식으로 만들어줘.
        4. 문제 난이도는 "{difficulty}" 수준으로 구성해줘.
        5. 각 문제에는 "문제 / 정답 / 해설"을 포함시켜 줘.
        6. 답변에는 취소선을 사용하지마
        7. 기본 문항 수는 기본으로 3개로 하고, 사용자 지정한다면 요청하는 문항 개수를 따라줘.
        8. 사용자 질문이 업로드된 파일과 무관할 경우 "대답할 수 없습니다"라고 답변해.
        9. 마지막 멘트는 "다른 문제를 추가로 내드릴까요?" 혹은 "난이도를 조정해드릴까요?"로 마무리해 줘.                                      
        """)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        parser = StrOutputParser()

        chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | prompt.partial(question_type=question_type, difficulty=difficulty)
            | llm
            | parser
        )

        response = chain.invoke(user_query)

        # === 3️⃣ 세션 메모리 업데이트 ===
        session_memory["last_question"] = user_query
        session_memory["last_difficulty"] = difficulty
        session_memory["last_file_path"] = temp_path

        return response.strip()

    except Exception as e:
        return f"❌ 문제 생성 중 오류가 발생했습니다: {str(e)}"
