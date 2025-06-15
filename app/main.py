import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 환경 변수 로드
load_dotenv()

# API 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing!")

# 디렉토리 설정
VECTOR_DIR = "data/vectorstore"

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 벡터스토어, 임베딩, QA 체인 미리 초기화
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local(
    VECTOR_DIR,
    embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 검색 문서 수 줄여 속도 개선
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3, model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 요청 데이터 모델
class QueryRequest(BaseModel):
    query: str

# 질문 처리 API
@app.post("/query")
async def ask_pdf_bot(request: QueryRequest):
    try:
        result = qa_chain({"query": request.query})
        answer = result["result"]
        sources = [doc.metadata.get("source", "N/A") for doc in result["source_documents"]]

        return JSONResponse(content={
            "question": request.query,
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 헬스 체크 API
@app.get("/health")
def health():
    return {"status": "ok"}
