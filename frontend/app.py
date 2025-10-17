import streamlit as st
from streamlit_chat import message
from io import BytesIO
from question_generation import generate_questions  # 실제 문제 생성기 함수 호출

st.set_page_config(page_title="스마트팜 어시스턴트", layout="centered")
st.title("스마트팜 어시스턴트")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요? 😊\n\n다음 중 하나를 선택해주세요:"}
    ]
    st.session_state.selected_function = None
    st.session_state.user_input = ""

# 기능 선택 함수
def select_function(option):
    st.session_state.selected_function = option
    st.session_state.messages.append({
        "role": "user",
        "content": f"{option} 기능을 사용할게요."
    })

# 채팅 UI 출력
for idx, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=(msg["role"] == "user"), key=f"chat_{idx}")

# 기능 선택 탭
if st.session_state.selected_function is None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 문서 요약", key="btn_summary"):
            select_function("문서 요약")
        if st.button("❓ 재배기술 QnA", key="btn_qna"):
            select_function("재배기술 QnA")
    with col2:
        if st.button("📝 퀴즈 생성기", key="btn_quiz"):
            select_function("퀴즈 생성기")
        if st.button("📚 자료 추천", key="btn_recommend"):
            select_function("자료 추천")

# 기능 입력 영역
if st.session_state.selected_function:
    st.divider()
    st.subheader(f"🔍 {st.session_state.selected_function} 기능")

    # ⛔ 기능 취소 버튼 (key 명시!)
    if st.button("🔙 기능 선택 취소", key="cancel_function"):
        st.session_state.selected_function = None
        st.session_state.messages.append({
            "role": "user",
            "content": "기능 선택을 취소할게요."
        })
        st.rerun()

    # 📝 퀴즈 생성기
    if st.session_state.selected_function == "퀴즈 생성기":
        question_type = st.radio(
            "퀴즈 유형을 선택해주세요",
            ["객관식", "주관식"],
            horizontal=True,
            key="question_type"
        )
        uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요", type="pdf", key="quiz_pdf_upload")
        user_query = st.text_input("생성할 퀴즈 유형에 대한 질문을 입력해주세요", placeholder="예: 토마토 재배 적정 온도에 대해 주관식 문제를 만들어줘", key="quiz_input")

        if st.button("퀴즈 생성 요청", key="generate_quiz"):
            if uploaded_file and user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                pdf_bytes = BytesIO(uploaded_file.read())
                with st.spinner("퀴즈를 생성 중입니다..."):
                    try:
                        result = generate_questions(user_query, question_type, pdf_bytes)
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        st.rerun()
                    except Exception as e:
                        st.error(f"퀴즈 생성 중 오류 발생: {e}")
            else:
                st.warning("PDF 파일과 질문을 모두 입력해주세요!")

    # 📄 문서 요약, ❓ QnA, 📚 추천
    else:
        uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요", type="pdf", key="doc_input_upload")
        user_query = st.text_input("질문을 입력해주세요", placeholder="예: 토마토 정식 이후 관리법을 요약해줘", key="doc_input_query")

        if st.button("요청 실행", key="submit_other"):
            if uploaded_file and user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.spinner("AI가 응답 중입니다..."):
                    result = "요약 결과: 토마토는 정식 후 1주일 간 수분 조절이 중요합니다."
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    st.rerun()
            else:
                st.warning("PDF 파일과 질문을 모두 입력해주세요!")
