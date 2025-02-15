import os
import torch
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
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

# Streamlit 앱 제목 설정
st.title("DeepSeek R1 & ollama를 활용한 RAG 시스템 구축")

# PDF 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file is not None:
    # 파일명에서 확장자를 제외한 이름 추출
    file_name = Path(uploaded_file.name).stem
    vector_store_dir = f"vector_stores/{file_name}"

    # 벡터 스토어가 이미 존재하는지 확인
    if os.path.exists(vector_store_dir):
        st.write("저장된 벡터 스토어를 불러옵니다.")
        # 기존 벡터 스토어 로드 - allow_dangerous_deserialization 파라미터 추가
        embedder = HuggingFaceEmbeddings()
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
        embedder = HuggingFaceEmbeddings()
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
    # llm = OllamaLLM(model="deepseek-r1:14b")
    llm = OllamaLLM(model="sisaai/sisaai-llama3.1:latest")

    # 시스템 프롬프트 정의
    system_prompt = (
        "주어진 문맥을 참고하여 질문에 답하세요. "
        "답을 모를 경우, '모르겠습니다'라고만 답하고 스스로 답을 만들지 마세요. "
        "답변은 최대 5문장으로 간결하게 작성하세요. "
        "최종 답변은 무조건 한국어(korean)으로 작성해주세요"
        "문맥: {context}"
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
