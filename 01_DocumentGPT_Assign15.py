import os
import streamlit as st

from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


# 파일 분리 (함수들)
from utils.functions.chat import ChatMemory, ChatCallbackHandler

# Streamlit 페이지 설정
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
    layout="wide",
)

# 세션 상태 초기화
for key, default in [
    ("messages", []),
    ("api_key", None),
    ("api_key_check", False),
    ("openai_model", "선택해주세요"),
    ("openai_model_check", False),
    ("file_check", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# 페이지 제목 및 설명
st.title("DocumentGPT")

if not (
    st.session_state["api_key_check"]
    and st.session_state["openai_model_check"]
    and st.session_state["file_check"]
):
    st.markdown(
        """
        안녕하세요! 이 페이지는 문서를 읽어주는 AI입니다.😄 
        
        문서를 업로드하고 질문을 하면 문서에 대한 답변을 해줍니다.
        """
    )

    if not st.session_state["file_check"]:
        st.warning("문서를 업로드해주세요.")
    else:
        st.success("😄문서가 업로드되었습니다.😄")
    if not st.session_state["api_key_check"]:
        st.warning("API_KEY를 넣어주세요.")
    else:
        st.success("😄API_KEY가 저장되었습니다.😄")
    if not st.session_state["openai_model_check"]:
        st.warning("모델을 선택해주세요.")
    else:
        st.success("😄모델이 선택되었습니다.😄")
else:
    st.success("😄API_KEY와 모델이 저장되었습니다.😄")


class FileController:
    # 파일 임베딩 함수
    @staticmethod
    @st.cache_resource(show_spinner="Embedding file...")
    def embed_file(file):
        os.makedirs("./.cache/files", exist_ok=True)
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.read())

        cache_dir = LocalFileStore(f"./.cache/embeddings/open_ai/{file.name}")
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=["\n\n", ".", "?", "!"],
            chunk_size=1000,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["api_key"])
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        return vectorstore.as_retriever()

    # 문서 포맷팅 함수
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)


# 사이드바 설정
with st.sidebar:
    st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
        key="file",
    )
    if st.session_state["file_check"]:
        st.success("😄문서가 업로드되었습니다.😄")
    else:
        st.warning("문서를 업로드해주세요.")

    st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        key="api_key",
    )

    if st.session_state["api_key_check"]:
        st.success("😄API_KEY가 저장되었습니다.😄")
    else:
        st.warning("API_KEY를 넣어주세요.")

    st.selectbox(
        "OpenAI Model을 골라주세요.",
        key="openai_model",
    )

    if st.session_state["openai_model_check"]:
        st.success("😄모델이 선택되었습니다.😄")
    else:
        st.warning("모델을 선택해주세요.")

    st.write(
        """
        Made by hary.
        
        Github
        https://github.com/lips85/GPT_hary

        streamlit
        https://hary-gpt.streamlit.app/
        """
    )

# 메인 로직
if (
    st.session_state["api_key_check"]
    and st.session_state["file_check"]
    and st.session_state["openai_model_check"]
):
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        model=st.session_state["openai_model"],
        openai_api_key=st.session_state["api_key"],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI that reads documents for me. Please answer based on the document given below. 
                If the information is not in the document, answer the question with "The required information is not in the document." Never make up answers.
                Please answer in the questioner's language 
                
                Context : {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    retriever = (
        FileController.embed_file(st.session_state["file"])
        if st.session_state["file_check"]
        else None
    )
    if retriever:
        ChatMemory.send_message("I'm ready! Ask away!", "ai", save=False)
        ChatMemory.paint_history()
        message = st.chat_input("Ask anything about your file...")

        if message:

            ChatMemory.send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(FileController.format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            try:
                with st.chat_message("ai"):
                    chain.invoke(message)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("OPENAI_API_KEY or 모델 선택을 다시 진행해주세요.")

    else:
        st.session_state["messages"] = []
