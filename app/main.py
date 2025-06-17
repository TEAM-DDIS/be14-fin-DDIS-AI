import os
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from .prompts import friendly_policy_prompt, friendly_empathy_prompt, fallback_hr_prompt

# 1. Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing!")

# 2. Constants
VECTOR_DIR = "data/vectorstore"
TRIVIAL_KEYWORDS = ["안녕", "하이", "ㅎㅇ", "반가워", "고마워", "감사", "넵", "응", "ㅋㅋ", "ㅎㅎ"]

# 3. App & CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Load vectorstore
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(
    VECTOR_DIR,
    embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "score_threshold": 0.75}
)

# 5. LLM & QA chain
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0.3,
    streaming=True,
    model="gpt-3.5-turbo-0125"
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": friendly_policy_prompt}
)

# 6. Input model
class QueryRequest(BaseModel):
    query: str

# 7. 감성 판별
def is_trivial(query: str) -> bool:
    return any(query.strip().lower().startswith(k) for k in TRIVIAL_KEYWORDS)

# 8. 무의미 질의 필터링
def is_garbage_query(query: str) -> bool:
    q = query.strip().lower()
    return len(q) < 2 or all(c in "ㅋㅎㄱㅜㅠ" for c in q) or not any(c.isalnum() for c in q)

# 9. Streaming 응답 생성기
async def stream_llm_response(prompt: str) -> AsyncGenerator[str, None]:
    try:
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content
    except Exception:
        yield "\n[오류] 응답 중 문제가 발생했습니다."

# 10. Streaming 질의 응답 API
@app.post("/query-stream")
async def query_stream(request: QueryRequest):
    query = request.query.strip()

    # 감성 응답
    if is_trivial(query):
        prompt = friendly_empathy_prompt.format(query=query)
        return StreamingResponse(stream_llm_response(prompt), media_type="text/plain")

    # 무의미 질의
    if is_garbage_query(query):
        return StreamingResponse(
            stream_llm_response("죄송하지만 질문을 좀 더 구체적으로 입력해 주실 수 있나요? 😊"),
            media_type="text/plain"
        )

    try:
        # 직접 문서 검색 (점수 포함)
        # docs_with_scores = retriever.vectorstore.similarity_search_with_score(query, k=5)
        # high_score_docs = [doc for doc, score in docs_with_scores if score >= 0.75]

        # if high_score_docs:
        #     context = "\n".join([doc.page_content for doc in high_score_docs])
        #     prompt = friendly_policy_prompt.format(context=context, question=query)
        # else:
        #     prompt = fallback_hr_prompt.format(question=query)

        # return StreamingResponse(stream_llm_response(prompt), media_type="text/plain")
    
        result = qa_chain.invoke({"query": query})
        sources = result["source_documents"]

        if sources:
            context = "\n".join([doc.page_content for doc in sources])
            prompt = friendly_policy_prompt.format(context=context, question=query)
        else:
            prompt = fallback_hr_prompt.format(question=query)

        return StreamingResponse(stream_llm_response(prompt), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
