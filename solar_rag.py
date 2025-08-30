import time
import os
import base64
import uuid
import tempfile
from typing import Dict, List, Any, Optional
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import streamlit as st
from langsmith import Client
import os

# .env 파일에서 upstage key 받아오기
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")

# LangSmith 키
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# LangSmith 기본 트레이싱 설정(최소)
os.environ["LANGCHAIN_TRACING_V2"] = "true"           # 트레이싱 ON
os.environ["LANGCHAIN_PROJECT"]     = "solar-rag"     # 프로젝트명(없으면 default)
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"  # US 기본

client = Client()  # env로부터 자동 설정
# # 프로젝트 목록이 보이면 인증 OK
projects = list(client.list_projects())[:3]
print("LangSmith projects (sample):", [p.name for p in projects] or "(none)")

# 세션 상태 초기화
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

# 세션 ID 설정
session_id = st.session_state.id
client = None

# 채팅 초기화 함수 정의
def reset_chat() -> None:
    """나눴던 대화와 불러온 문서 초기화하는 함수
    """
    st.session_state.messages = []
    st.session_state.context = None

# 읽어온 PDF 를 보여주는 함수
def display_pdf(file) -> None:
    """PDF 파일을 받아와서 디스플레이 해주는 함수
    """
    st.markdown ("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf" style="height:100vh; width:100%"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:
    st.header(f"Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf`file",type="pdf")

    if uploaded_file:
        print(uploaded_file)
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir,uploaded_file.name)
                print("file path:", file_path)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                st.write("Indexing your document ...")

                # 위에서 key 와 temp_dir 생겼는지 체크
                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        print("temp_dir:",temp_dir)
                        loader = PyPDFLoader(file_path)
                    else:
                        st.error('Colud not find the file you uploaded, please check again ...')
                        st.stop()
                
                    pages = loader.load_and_split()

                    vectorstore = FAISS.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))

                    retriever = vectorstore.as_retriever(k=2)

                    llm = ChatUpstage()

                    contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""

                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    
                    # 이전 대화를 기억하는 리트리버 생성
                    history_aware_retriever = create_history_aware_retriever(
                        llm, retriever, contextualize_q_prompt
                    )

                    qa_system_prompt = """질문-답변 업무를 돕는 보조원입니다. 질문에 답하기 위해 검색된 내용을 사용하세요. 답을 모르면 모른다고 말하세요. 답변은 세 문장 이내로 간결하게 유지하세요.
                    ## 답변 예시
                    📍답변 내용:
                    📍증거:
                    {context}"""

                    qa_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", qa_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                    
                    st.success("Ready to Chat!")
                    display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occuered : {e}")
            st.stop()


# 웹사이트 제목 작성
st.title("Solar LLM Chatbot")

# 메세지 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메세지 표시
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# 기록하는 대화의 최대 길이를 설정
MAX_MESSAGES_BEFORE_DELETION = 8

# 유저입력 처리
if prompt := st.chat_input("질문을 입력하세요!"):
    # 이전 대화의 길이 확인
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        del st.session_state.messages[0]
        del st.session_state.messages[0]

    st.session_state.messages.append(
        {"role": "user","content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 의 답변을 받아서 session state에 저장하고, 보여도 줘야함
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_reponse =""

        result = rag_chain.invoke(
            {'input':prompt, 'chat_history': st.session_state.messages}
        )
        # print(result)
        with st.expander("불러온 문서"):
            st.write(result['context'])

        for chunk in result['answer'].split(" "):
            full_reponse += chunk + " "
            message_placeholder.markdown(full_reponse)
    
    st.session_state.messages.append(
        {"role": "assistant","content": full_reponse})