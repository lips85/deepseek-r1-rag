import os
import torch
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path

# models
from models import LLM_options, EMBEDDING_options


# 채팅 관련 클래스 추가
class ChatMemory:
    @staticmethod
    def save_message(message, role):
        st.session_state["messages"].append({"message": message, "role": role})

    @staticmethod
    def send_message(message, role, save=True):
        with st.chat_message(role):
            st.markdown(message)
        if save:
            ChatMemory.save_message(message, role)

    @staticmethod
    def paint_history():
        for message in st.session_state["messages"]:
            ChatMemory.send_message(message["message"], message["role"], save=False)


# 색상 팔레트 정의

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


# INITAILIZE
for key, default in [
    ("uploaded_file", False),
    ("messages", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# Streamlit 앱 제목 설정
st.title("DeepSeek R1(32B) RAG")

# PDF 파일 업로드

with st.sidebar:
    st.markdown("## 1.LLM 모델을 골라주세요.")
    model = st.selectbox(
        label="## 1.LLM 모델을 골라주세요.",
        label_visibility="collapsed",
        options=LLM_options,
    )
    st.write("## 2.embedding 모델을 골라주세요.")
    select_embedder = st.selectbox(
        "2.embedding 모델을 골라주세요.",
        label_visibility="collapsed",
        options=EMBEDDING_options,
    )

    if st.session_state["uploaded_file"] is False:
        st.write("## 3.PDF 파일을 업로드하세요")
        st.file_uploader(
            "3.PDF 파일을 업로드하세요",
            type="pdf",
            key="file",
            label_visibility="collapsed",
        )

if st.session_state["file"] is not None:
    st.success("upload finished")
    uploaded_file = st.session_state["file"]
    # 파일명에서 확장자를 제외한 이름 추출
    file_name = Path(uploaded_file.name).stem
    vector_store_dir = f"vector_stores/{select_embedder}/{file_name}"

    # 벡터 스토어가 이미 존재하는지 확인
    if os.path.exists(vector_store_dir):

        # 기존 벡터 스토어 로드 - allow_dangerous_deserialization 파라미터 추가
        st.write("저장된 벡터 스토어를 불러옵니다.")

        if select_embedder == "HuggingFaceEmbeddings":
            embedder = HuggingFaceEmbeddings()
        else:
            embedder = OllamaEmbeddings(model=select_embedder)

        vector = FAISS.load_local(
            vector_store_dir, embedder, allow_dangerous_deserialization=True
        )
        retriever = vector.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
    else:
        with st.spinner("새로운 벡터 스토어를 생성합니다."):
            # 임시 파일로 저장
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

        # PDF 로더 초기화
        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()

        # 문서 분할기 초기화
        if select_embedder == "HuggingFaceEmbeddings":
            embedder = HuggingFaceEmbeddings()
        else:
            embedder = OllamaEmbeddings(model=select_embedder)

        text_splitter = SemanticChunker(embedder)
        documents = text_splitter.split_documents(docs)

        # 벡터 스토어 생성 및 임베딩 추가
        vector = FAISS.from_documents(documents, embedder)

        # 벡터 스토어 저장
        os.makedirs(vector_store_dir, exist_ok=True)
        vector.save_local(vector_store_dir)

        retriever = vector.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

    # LLM 정의 (OllamaLLM 대신 ChatOllama 사용)
    llm = ChatOllama(
        model=model,
        temperature=0.5,
        streaming=True,  # 스트리밍 명시적으로 활성화
    )

    # 시스템 프롬프트 정의
    system_prompt = """
        주어진 문맥을 참고하여 질문에 답하세요.
        답을 모를 경우, '모르겠습니다'라고만 답하고 스스로 답을 만들지 마세요.
        최종 답변은 무조건 한국어(korean)으로 작성해주세요
        문맥: {context}
        """

    # ChatPromptTemplate 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 문서 결합 체인 생성
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # 문서 검색 기반 QA 체인 생성
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 데모 메시지 추가 및 채팅 기록 표시
    if not st.session_state["messages"]:
        # 메시지를 세션에만 저장하고 화면에는 표시하지 않음
        st.session_state["messages"].append(
            {"message": "PDF 문서와 관련된 질문을 입력해주세요!", "role": "ai"}
        )

    # 세션에 저장된 모든 메시지 표시
    ChatMemory.paint_history()

    # 채팅 입력 사용
    user_input = st.chat_input("PDF와 관련된 질문을 입력하세요:")

    # 사용자 입력 처리
    if user_input:
        # 사용자 메시지 표시
        ChatMemory.send_message(user_input, "human")

        with st.chat_message("ai"):
            try:
                # 메시지 박스 준비
                message_placeholder = st.empty()
                full_response = ""

                # 직접 검색하고 채팅 모델에 전달
                docs = retriever.invoke(user_input)
                context = "\n\n".join(doc.page_content for doc in docs)

                # 메시지 생성
                messages = [
                    ("system", system_prompt.replace("{context}", context)),
                    ("human", user_input),
                ]

                # 직접 ChatOllama로 스트리밍
                for chunk in llm.stream(messages):
                    if hasattr(chunk, "content"):
                        content = chunk.content
                        if content:
                            full_response += content
                            message_placeholder.markdown(full_response)

                # 응답 저장
                if full_response:
                    ChatMemory.save_message(full_response, "ai")
                else:
                    # 응답이 없는 경우 기본 메시지 표시
                    st.warning("답변을 생성할 수 없습니다.")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                import traceback

                st.error(f"상세 오류: {traceback.format_exc()}")
else:
    st.write("진행하려면 PDF 파일을 업로드하세요")
