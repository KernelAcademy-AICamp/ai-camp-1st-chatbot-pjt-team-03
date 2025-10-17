from anthropic import Anthropic
from typing import Dict, Optional


class QnASystem:
    """Q&A 전문 모듈 - 배경지식과 사용자 PDF를 종합하여 답변"""

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def answer_with_knowledge(self, question: str, knowledge_vectorstore, user_pdf_vectorstore, top_k: int = 5) -> Dict:
        """
        Q&A (배경지식 + 사용자 PDF 모두 활용)

        Args:
            question: 사용자 질문
            knowledge_vectorstore: 배경지식 벡터스토어 객체
            user_pdf_vectorstore: 사용자 PDF 벡터스토어 객체
            top_k: 각 인덱스에서 검색할 청크 개수

        Returns:
            {"answer": "답변", "knowledge_sources": [...], "pdf_sources": [...]}
        """
        # 배경지식에서 검색
        knowledge_results = knowledge_vectorstore.search(question, top_k=top_k) if knowledge_vectorstore else []
        
        # 사용자 PDF에서 검색
        pdf_results = user_pdf_vectorstore.search(question, top_k=top_k) if user_pdf_vectorstore else []

        if not knowledge_results and not pdf_results:
            return {
                "answer": "관련 정보를 찾을 수 없습니다. 문서를 업로드하거나 질문을 다시 작성해주세요.", 
                "knowledge_sources": [],
                "pdf_sources": []
            }

        # 컨텍스트 구성
        context_parts = []
        
        if knowledge_results:
            context_parts.append("=== 배경지식 ===")
            context_parts.extend([
                f"[배경지식 {i+1}]\n{chunk['text']}"
                for i, (chunk, score) in enumerate(knowledge_results)
            ])
        
        if pdf_results:
            context_parts.append("=== 업로드된 문서 ===")
            context_parts.extend([
                f"[문서 {i+1}]\n{chunk['text']}"
                for i, (chunk, score) in enumerate(pdf_results)
            ])

        context = "\n\n".join(context_parts)

        prompt = f"""당신은 학습을 돕는 AI 튜터입니다.
배경지식과 업로드된 문서를 모두 참고하여 학생의 질문에 답변하세요.

<참고 자료>
{context}
</참고 자료>

<학생 질문>
{question}
</학생 질문>

답변 규칙:
1. 배경지식과 문서 내용을 종합하여 답변
2. 자료에 없는 내용은 "해당 내용은 자료에서 찾을 수 없습니다"라고 명시
3. 배경지식과 문서 내용이 다르면 그 차이점도 설명
4. 쉽고 명확하게 설명하며 필요시 예시 추가

답변:"""

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.content[0].text

        # 출처 정보 분리
        knowledge_sources = [
            {
                "text": chunk["text"][:200] + "...",
                "score": f"{score:.2f}",
                "index": chunk["index"],
                "type": "knowledge"
            }
            for chunk, score in knowledge_results
        ]

        pdf_sources = [
            {
                "text": chunk["text"][:200] + "...",
                "score": f"{score:.2f}",
                "index": chunk["index"],
                "type": "user_pdf"
            }
            for chunk, score in pdf_results
        ]

        return {
            "answer": answer, 
            "knowledge_sources": knowledge_sources,
            "pdf_sources": pdf_sources
        }

    def answer_with_knowledge_only(self, question: str, knowledge_vectorstore, top_k: int = 5) -> Dict:
        """
        배경지식만으로 Q&A

        Args:
            question: 사용자 질문
            knowledge_vectorstore: 배경지식 벡터스토어 객체
            top_k: 검색할 청크 개수

        Returns:
            {"answer": "답변", "sources": [...]}
        """
        if not knowledge_vectorstore:
            return {
                "answer": "배경지식이 준비되지 않았습니다.", 
                "sources": []
            }

        results = knowledge_vectorstore.search(question, top_k=top_k)

        if not results:
            return {
                "answer": "배경지식에서 관련 정보를 찾을 수 없습니다.", 
                "sources": []
            }

        context = "\n\n".join(
            [
                f"[배경지식 {i+1}]\n{chunk['text']}"
                for i, (chunk, score) in enumerate(results)
            ]
        )

        prompt = f"""당신은 학습을 돕는 AI 튜터입니다.
배경지식을 참고하여 학생의 질문에 답변하세요.

<배경지식>
{context}
</배경지식>

<학생 질문>
{question}
</학생 질문>

답변 규칙:
1. 배경지식 내용을 기반으로 정확하게 답변
2. 자료에 없는 내용은 "해당 내용은 자료에서 찾을 수 없습니다"라고 명시
3. 쉽고 명확하게 설명하며 필요시 예시 추가

답변:"""

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.content[0].text

        sources = [
            {
                "text": chunk["text"][:200] + "...",
                "score": f"{score:.2f}",
                "index": chunk["index"],
                "type": "knowledge"
            }
            for chunk, score in results
        ]

        return {"answer": answer, "sources": sources}

    def answer_with_pdf_only(self, question: str, user_pdf_vectorstore, top_k: int = 5) -> Dict:
        """
        사용자 PDF만으로 Q&A

        Args:
            question: 사용자 질문
            user_pdf_vectorstore: 사용자 PDF 벡터스토어 객체
            top_k: 검색할 청크 개수

        Returns:
            {"answer": "답변", "sources": [...]}
        """
        if not user_pdf_vectorstore:
            return {
                "answer": "업로드된 문서가 없습니다. PDF를 먼저 업로드해주세요.", 
                "sources": []
            }

        results = user_pdf_vectorstore.search(question, top_k=top_k)

        if not results:
            return {
                "answer": "업로드된 문서에서 관련 정보를 찾을 수 없습니다.", 
                "sources": []
            }

        context = "\n\n".join(
            [
                f"[문서 {i+1}]\n{chunk['text']}"
                for i, (chunk, score) in enumerate(results)
            ]
        )

        prompt = f"""당신은 학습을 돕는 AI 튜터입니다.
업로드된 문서를 참고하여 학생의 질문에 답변하세요.

<업로드된 문서>
{context}
</업로드된 문서>

<학생 질문>
{question}
</학생 질문>

답변 규칙:
1. 문서 내용을 기반으로 정확하게 답변
2. 문서에 없는 내용은 "자료에서 해당 내용을 찾을 수 없습니다"라고 답변
3. 쉽고 명확하게 설명하며 필요시 예시 추가

답변:"""

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.content[0].text

        sources = [
            {
                "text": chunk["text"][:200] + "...",
                "score": f"{score:.2f}",
                "index": chunk["index"],
                "type": "user_pdf"
            }
            for chunk, score in results
        ]

        return {"answer": answer, "sources": sources}
