# chatbot.py
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from app.prompts import friendly_policy_prompt

load_dotenv()

VECTOR_DIR = "data/vectorstore"

# 벡터스토어 불러오기
vectorstore = FAISS.load_local(
    folder_path=VECTOR_DIR,
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM 설정
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": friendly_policy_prompt
    }
)

# 인터랙티브 모드
while True:
    query = input("\n질문을 입력하세요 (종료: 'exit'): ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain(query)
    print("\n답변:", result["result"])

    print("\n사용된 문서:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("fileName"), "→", doc.metadata.get("tags"))
