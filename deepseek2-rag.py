import os
import torch
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path

# 색상 팔레트 정의
primary_color = "#1E90FF"  # 기본 색상
secondary_color = "#FF6347"  # 보조 색상
background_color = "#F5F5F5"  # 배경 색상
text_color = "#4561e9"  # 텍스트 색상

# 사용자 정의 CSS 적용
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    </style>
""",
    unsafe_allow_html=True,
)
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


# INITAILIZE
for key, default in [
    ("uploaded_file", False)
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
    options=["deepseek-r1:32b", "deepseek-r1:70b","sisaai/sisaai-llama3.1:latest", "benedict/linkbricks-llama3.1-korean:70b"],
)
    st.write("## 2.embedding 모델을 골라주세요.")
    select_embedder = st.selectbox(
    "2.embedding 모델을 골라주세요.",
    label_visibility="collapsed",
    options=("snowflake-arctic-embed2","HuggingFaceEmbeddings", "nomic-embed-text:latest"),
)

    if st.session_state['uploaded_file'] is False:
        st.write("## 3.PDF 파일을 업로드하세요")
        st.file_uploader("3.PDF 파일을 업로드하세요", type="pdf", key="file", label_visibility="collapsed")

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

        if select_embedder is "HuggingFaceEmbeddings":
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
        if select_embedder is "HuggingFaceEmbeddings":
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

    # LLM 정의
    llm = OllamaLLM(model=model, temperature=0.2)

    # 시스템 프롬프트 정의
    system_prompt = (
        """
        주어진 문맥을 참고하여 질문에 답하세요.
        답을 모를 경우, '모르겠습니다'라고만 답하고 스스로 답을 만들지 마세요.
        최종 답변은 무조건 한국어(korean)으로 작성해주세요
        문맥: {context}
        """
    )

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

    # 검색 기반 QA 체인 생성
    user_input = st.text_input("PDF와 관련된 질문을 입력하세요:")

    # 사용자 입력 처리
    if user_input:
        with st.spinner("처리 중 ..."):
            response = rag_chain.invoke({"input": user_input})
            st.write("응답:")
            st.write(response.get("answer", "응답을 처리할 수 없습니다."))
else:
    st.write("진행하려면 PDF 파일을 업로드하세요")
