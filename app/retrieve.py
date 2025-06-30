import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()


def get_retriever(
    persist_dir: str = "chroma_local_policies",
    model_name: str = "text-embedding-ada-002",
    k: int = 3,
    fetch_k: int = 10
):
    """
    저장된 ChromaDB에서 Retriever를 생성하여 반환합니다.

    :param persist_dir: Chroma 인덱스 디렉토리
    :param model_name: OpenAI 임베딩 모델명
    :param k: 최종 반환할 유사 문서 개수
    :param fetch_k: 내부 탐색 시 볼 후보 벡터 개수
    """
    # 임베딩 함수 생성
    embeddings = OpenAIEmbeddings(
        model=model_name,
        chunk_size=1,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Chroma 벡터 스토어 로드
    chroma = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Retriever 생성
    retriever = chroma.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k
        }
    )
    # retriever = chroma.as_retriever(
    #     search_kwargs={"k": k}
    # )
    return retriever


def get_qa_chain(retriever):
    llm = ChatOpenAI(
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

def main():
    retriever = get_retriever()
    qa_chain  = get_qa_chain(retriever)

    print("=== 정책 문서 QA 시스템 ===")
    while True:
        query = input("질문을 입력하세요 (종료: exit): ")
        if query.strip().lower() == "exit":
            break

        # 🔹 RAG 답변 생성
        result = qa_chain({"query": query})
        answer = result["result"]                # 생성된 답변
        docs   = result["source_documents"]      # 참조된 문서들

        # 출력
        print(f"\n[Query] {query}\n")
        print(f"[Answer]\n{answer}\n")
        print("[Sources]")
        for i, doc in enumerate(docs, 1):
            snippet = doc.page_content.replace("\n", " ")[:200]
            source  = doc.metadata.get("source", "Unknown")
            print(f" ({i}) {snippet}...  [{source}]")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()