# -------------------------------------------------------------------------
# 참고: 이 코드의 일부는 다음 GitHub 리포지토리에서 참고하였습니다:
# https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407
# 해당 리포지토리의 라이센스에 따라 사용되었습니다.
# -------------------------------------------------------------------------

# Streamlit 라이브러리 및 기타 필요한 라이브러리 임포트
import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from utils import load_model, set_memory, initialize_chain, generate_message

# 애플리케이션 제목 설정
st.title("페르소나 챗봇")
st.markdown("<br>", unsafe_allow_html=True)  # 브라우저에 줄 바꿈을 삽입합니다.

# 사용자로부터 캐릭터를 선택받기 위한 드롭다운 메뉴 설정
character_name = st.selectbox(
    "**캐릭터를 골라줘!**",
    ("trump", "heavy_grandma", "IU"),
    index=0,
    key="character_name_select",
)
# 선택된 캐릭터 이름을 세션 상태에 저장
st.session_state.character_name = character_name

# 사용자로부터 모델을 선택받기 위한 드롭다운 메뉴 설정
model_name = st.selectbox(
    "**모델을 골라줘!**",
    ("solar-mini", "solar-pro2"),
    index=0,
    key="model_name_select",
)
# 선택된 모델 이름을 세션 상태에 저장
st.session_state.model_name = model_name

# 세션 상태에서 채팅 시작 여부를 확인 및 초기화
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
    st.session_state.memory = None
    st.session_state.chain = None


# 채팅을 시작하는 함수 정의
def start_chat() -> None:
    """
    선택된 모델과 캐릭터를 기반으로 채팅을 시작합니다.

    Streamlit 세션 상태를 사용하여 사용자가 선택한 모델과 캐릭터 이름에 따라
    언어 모델을 로드하고 대화 메모리를 설정하며, LLM 체인을 초기화합니다.
    """
    llm = load_model(st.session_state.model_name)  # 선택된 모델을 로드합니다.
    st.session_state.chat_started = True  # 채팅 시작 상태를 True로 설정합니다.
    st.session_state.memory = set_memory()  # 메모리를 초기화합니다.
    st.session_state.chain = initialize_chain(
        llm, st.session_state.character_name, st.session_state.memory
    )  # 체인을 초기화합니다.


# "Start Chat" 버튼을 클릭했을 때 start_chat 함수를 호출
if st.button("Start Chat"):
    start_chat()

st.markdown("<br>", unsafe_allow_html=True)  # 브라우저에 줄 바꿈을 삽입합니다.

# 채팅이 시작된 경우
if st.session_state.chat_started:
    if st.session_state.memory is None or st.session_state.chain is None:
        start_chat()  # 메모리나 체인이 초기화되지 않은 경우 다시 초기화합니다.

    # 메모리에 저장된 모든 메시지를 화면에 표시
    for message in st.session_state.memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            role = "user"  # HumanMessage는 "user" 역할로 설정
        elif isinstance(message, AIMessage):
            role = "assistant"  # AIMessage는 "assistant" 역할로 설정
        else:
            continue
        with st.chat_message(role):  # 해당 역할로 메시지를 화면에 표시
            st.markdown(message.content)

    # 사용자가 입력한 새로운 메시지를 처리
    if prompt := st.chat_input():
        with st.chat_message("user"):  # 사용자 메시지를 화면에 표시
            st.markdown(prompt)

        with st.chat_message("assistant"):  # AI 응답을 화면에 표시
            message_placeholder = st.empty()  # 응답을 표시할 자리 확보
            full_response = ""
            response_content = generate_message(
                st.session_state.chain, prompt
            )  # AI 응답 생성

            # 응답을 단어 단위로 나누어 점진적으로 화면에 표시
            for chunk in response_content.split():
                full_response += chunk + " "
                time.sleep(0.05)  # 각 단어 사이에 지연을 추가하여 애니메이션 효과
                message_placeholder.markdown(full_response + "▌")  # 진행 중 표시
            message_placeholder.markdown(full_response.strip())  # 최종 응답 표시
