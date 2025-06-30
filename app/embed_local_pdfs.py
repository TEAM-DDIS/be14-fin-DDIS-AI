import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

def embed_local_pdfs(
    dir_path: str = "./data/raw_policies",
    persist_dir: str = "chroma_local_policies",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Chroma:
    """
    로컬 디렉토리(dir_path) 아래의 모든 PDF를
    페이지 단위로 읽은 뒤, TextSplitter로 잘라서(chunks),
    ChromaDB 벡터 스토어를 만들고 persist_dir에 저장한 뒤 반환합니다.
    """
    all_docs = []

    # 1) PDF → 페이지 단위 Document
    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(dir_path, fname)
        print(f">>> Loading {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()  # 페이지별 Document 리스트

        # 2) 페이지별 Document를 텍스트 청크로 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        # 3) 메타데이터에 원본 파일명 추가
        for doc in chunks:
            doc.metadata["source"] = fname
            all_docs.append(doc)

    # 4) 임베딩 모델 (OpenAI)
    embedder = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=1,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 5) ChromaDB 인덱스 생성 및 저장
    vector_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embedder,
        persist_directory=persist_dir
    )
    vector_store.persist()
    return vector_store

if __name__ == "__main__":
    vs = embed_local_pdfs(
        chunk_size=500,     # 청크 하나당 최대 문자 수
        chunk_overlap=50    # 청크끼리 겹치는 문자 수
    )
    print("✅ PDF → 텍스트 청크 임베딩 및 ChromaDB 저장 완료!")