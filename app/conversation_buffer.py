import os
from dotenv import load_dotenv

# Embedding & Vector store
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Chains & Memory (최신 방식 사용)
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Retriever 설정
def get_retriever(
    persist_dir="chroma_local_policies",
    model_name="text-embedding-ada-002",
    k=3,
    fetch_k=10,
    lambda_mult=0.8,
):
    embeddings = OpenAIEmbeddings(
        model=model_name, openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    chroma = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return chroma.as_retriever(
        search_type="mmr",
        search_kwargs={"fetch_k": fetch_k, "k": k, "lambda_mult": lambda_mult},
    )

# LLM 정의
llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# 공유 메모리 정의
shared_memory = ConversationBufferMemory(return_messages=True)

# QA 체인 구성 (최신 방식)
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

retriever = get_retriever()

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="다음 문맥을 참고하여 질문에 답변하세요.\n\n{context}\n\n질문: {question}\n답변:",
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
)

# Chat 체인 구성 (최신 방식)
def chat_chain(input_text):
    messages = shared_memory.load_memory_variables({})["history"]
    messages.append(HumanMessage(content=input_text))
    response = llm.invoke(messages)
    shared_memory.save_context({"input": input_text}, {"output": response.content})
    return response.content

# Router 체인 구성 (최신 방식)
router_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "아래 사용자 질문을 보고,\n"
        "- 회사 정책·매뉴얼 조회가 필요하면 RETRIEVE\n"
        "- 단순 대화(인사·잡담·메타질의)면 CHAT\n"
        "만 출력하세요.\n\n"
        "질문: {question}\n"
        "출력:"
    ),
)

router_chain = router_prompt | llm | (lambda x: x.content.strip().upper())

# 분기 함수
def answer_with_routing(query: str):
    route = router_chain.invoke({"question": query})

    if route == "RETRIEVE":
        response = rag_chain.invoke(query)
        shared_memory.save_context({"input": query}, {"output": response.content})
        docs = retriever.get_relevant_documents(query)
        return response.content, docs
    else:
        resp = chat_chain(query)
        return resp, []

# 메인 실행
def main():
    print("=== 대화형 정책 문서 QA 시스템 ===")
    while True:
        user_q = input("입력: ")
        if user_q.strip().lower() == "exit":
            break

        answer, sources = answer_with_routing(user_q)

        print(f"\n[Query] {user_q}\n")
        print(f"[Answer]\n{answer}\n")
        if sources:
            print("[Sources]")
            for i, doc in enumerate(sources, 1):
                snippet = doc.page_content.replace("\n", " ")[:200]
                src = doc.metadata.get("source", "Unknown")
                print(f" ({i}) {snippet}...  [{src}]")
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
