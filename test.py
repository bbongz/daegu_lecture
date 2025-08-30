import time
import os
import base64
import uuid
import tempfile
from typing import Dict, List, Any, Optional
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_upstage import ChatUpstage

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# .env 파일에서 upstage key 받아오기
from dotenv import load_dotenv
load_dotenv(override=True, encoding="utf-8")
api_key = os.getenv("UPSTAGE_API_KEY")


file_path = "attention_is_all_you_need.pdf"


loader = PyPDFLoader(file_path)
                
pages = loader.load_and_split()
print("pages ready")
vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))
print("vectorstore ready")
retriever = vectorstore.as_retriever(k=2)
print("retriever ready")
llm = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"))
print("llm ready")

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

