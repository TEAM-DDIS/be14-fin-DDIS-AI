# app.py

import os
import json
import asyncio
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime

# Embedding & Vector store
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import time

# Chains & Memory (최신 방식)
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage
import httpx

load_dotenv()

app = FastAPI()

# CORS (프론트엔드 주소에 맞게 조정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트 주소
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——— 데이터 모델 ———
class QueryPayload(BaseModel):
    question: str

# ——— Retriever 정의 ———
def get_retriever(
    persist_dir="chroma_local_policies",
    model_name="text-embedding-ada-002",
    k=3,
    fetch_k=10,
    lambda_mult=0.85,
):
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    chroma = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return chroma.as_retriever(
        search_type="mmr",
        search_kwargs={"fetch_k": fetch_k, "k": k, "lambda_mult": lambda_mult},
    )

retriever = get_retriever()

# ——— LLM & Memory 정의 ———
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or ""
llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-4o-mini",
    openai_api_key=OPENAI_KEY,
)
shared_memory = ConversationBufferMemory(return_messages=True)

# RAG 체인 (비스트리밍 버전)
RAG_PROMPT = PromptTemplate(
    input_variables=[
        "currentDate",           
        "context", "question",
        "employeeName", "departmentName", "teamName",
        "rankName", "positionName", "jobName",
        "hireDate", "remainingVacationDays",
        "draftsCount", "rejectedCount", "teamMembers", "personalSchedules"
    ],
    template=(
        "당신은 회사의 공식 비서입니다.\n"
        "🗓️ 현재 날짜: {currentDate}\n"
        "아래 **문서**와 **사용자 정보**만을 근거로 답변하세요.\n"
        "절대 문서 외의 정보를 추가하거나 추측하지 마십시오.\n\n"
        "절대 알고 있는 사용자 정보외에 추가하거나 추측하지 마십시오.\n\n"

        "=== 사용자 정보 ===\n"
        "- 이름: {employeeName}\n"
        "- 부서: {departmentName}\n"
        "- 팀: {teamName}\n"
        "- 직급: {rankName}\n"
        "- 직책: {positionName}\n"
        "- 직무: {jobName}\n"
        "- 입사일: {hireDate}\n"
        "- 잔여 휴가일: {remainingVacationDays}일\n"
        "- 기안 문서: {draftsCount}건\n"
        "- 결재 대기 문서: {pendingCount}건\n"
        "- 반려된 문서: {rejectedCount}건\n"
        "- 팀원: {teamMembers}\n\n"

        "=== 개인 일정 ===\n"
        "{personalSchedules}\n\n"

        "=== 사내 규정 문서 ===\n"
        "{context}\n\n"

        "❓ **질문:** {question}\n\n"

        "💬 **답변 작성 규칙**\n"
        "1. 문서 내용에서 정확히 인용하되, 페이지나 섹션 번호를 괄호 안에 표시하세요.\n"
        "2. 문서에 없는 정보는 “문서에서 해당 내용을 찾을 수 없습니다.”라고만 답하십시오.\n"
        "3. 문장은 2~3개 단락으로 요약하고, 중요한 키워드는 굵게 표시하세요.\n\n"

        "**답변:**"
    )
)

# ——— 체인 조립 ———
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Chat 체인 (비스트리밍)
def chat_chain(input_text: str) -> str:
    history = shared_memory.load_memory_variables({})["history"]
    history.append(HumanMessage(content=input_text))
    response = llm.invoke(history)
    shared_memory.save_context({"input": input_text}, {"output": response.content})
    return response.content

def build_rag_chain(user_info: dict, llm_instance, current_date: str):
    # 사용자 정보 람다 모음
    user_vars = {
        "currentDate": RunnableLambda(lambda _: current_date),
        "employeeName": RunnableLambda(lambda _: user_info.get("employeeName","정보 없음")),
        "departmentName": RunnableLambda(lambda _: user_info.get("departmentName","정보 없음")),
        "teamName": RunnableLambda(lambda _: user_info.get("teamName","정보 없음")),
        "rankName": RunnableLambda(lambda _: user_info.get("rankName","정보 없음")),
        "positionName": RunnableLambda(lambda _: user_info.get("positionName","정보 없음")),
        "jobName": RunnableLambda(lambda _: user_info.get("jobName","정보 없음")),
        "hireDate": RunnableLambda(lambda _: user_info.get("hireDate","정보 없음")),
        "remainingVacationDays": RunnableLambda(lambda _: str(user_info.get("remainingVacationDays","정보 없음"))),
        "draftsCount": RunnableLambda(lambda _: str(user_info.get("draftsCount",0))),
        "pendingCount": RunnableLambda(lambda _: str(user_info.get("pendingCount",0))),
        "rejectedCount": RunnableLambda(lambda _: str(user_info.get("rejectedCount",0))),
        "teamMembers": RunnableLambda(lambda _: ", ".join(
            m["employeeName"] for m in user_info.get("teamMembers",[]) if m.get("employeeName")
        )),
        "personalSchedules": RunnableLambda(lambda _: "\n".join(
            f"- {s['scheduleDate']} {s['scheduleTime']} : {s['scheduleTitle']}"
            for s in user_info.get("personalSchedules",[])
        )),
    }

    spec = {"context": retriever | format_docs, "question": RunnablePassthrough()}
    spec.update(user_vars)
    return (spec | RAG_PROMPT) | llm_instance

# ─── USER_INFO 전용 제너레이터 ───
async def user_info_generator(user_info: dict, question: str):
    # 1. 시스템 메시지에 프로필 담기
    profile = format_user_profile(user_info)
    system_msg = SystemMessage(content=f"👤 사용자 정보:\n{profile}")
    user_msg = HumanMessage(content=question)
    today   = datetime.now().strftime("%Y년 %m월 %d일")
    date_msg = SystemMessage(content=f"🗓️ 현재 날짜: {today}")

    # 2. 스트리밍 LLM 세팅
    llm_stream = ChatOpenAI(
        temperature=0.0,
        model="gpt-4o-mini",
        streaming=True,
        openai_api_key=OPENAI_KEY,
    )

    # 3. 메시지 순서: [System, User]
    async for chunk in llm_stream.astream([system_msg, user_msg,date_msg]):
        yield chunk.content


_user_info_cache: dict[str, tuple[dict, float]] = {}
TTL = 300  # 5분
async def fetch_user_info(token: str) -> dict:
    entry = _user_info_cache.get(token)
    if entry and time.time() - entry[1] < TTL:
        return entry[0]
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:5000/chatbot",
            headers={"Authorization": token}
        )

        print(f"🔁 Spring 응답 코드: {response.status_code}")
        print(f"🔁 Spring 응답 내용: {response.text}")

        response.raise_for_status()
        data = response.json()
        _user_info_cache[token] = (data, time.time())
        return data
    

def format_user_profile(user_info: dict) -> str:
    lines = [
        f"이름: {user_info.get('employeeName', '정보 없음')}",
        f"부서: {user_info.get('departmentName', '정보 없음')}",
        f"팀: {user_info.get('teamName', '정보 없음')}",
        f"직급: {user_info.get('rankName', '정보 없음')}",
        f"직책: {user_info.get('positionName', '정보 없음')}",
        f"직무: {user_info.get('jobName', '정보 없음')}",
        f"입사일: {user_info.get('hireDate', '정보 없음')}",
        f"남은 연차: {user_info.get('remainingVacationDays', '정보 없음')}일",
        f"기안한 문서 수: {user_info.get('draftsCount', 0)}건",
        f"결재 대기 문서 수: {user_info.get('pendingCount', 0)}건",
        f"반려 문서 수: {user_info.get('rejectedCount', 0)}건",
    ]

    team = user_info.get("teamMembers", [])
    if team:
        team_names = [m["employeeName"] for m in team if m.get("employeeName")]
        lines.append(f"같은 팀 팀원: {', '.join(team_names)}")

    schedules = user_info.get("personalSchedules", [])
    if schedules:
        formatted_schedules = "\n".join([
            f"- {s['scheduleDate']} {s['scheduleTime']} : {s['scheduleTitle']}" for s in schedules
        ])
        lines.append(f"\n📆 개인 일정:\n{formatted_schedules}")

    return "\n".join(lines)

router_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "🔎 **사내 규정·정책·절차 관련 질문이 아니면 반드시 CHAT으로 분류하세요.**\n"
        "❗️정책·규정·프로세스·휴가·근태·인사발령·복지 등 회사 내부 문서에 있는 내용만 RETRIEVE로 분류합니다.\n"
        "👤 개인정보 조회(내 연차, 내 기안 상태 등)는 USER_INFO로 분류합니다.\n"
        "💬 그 외 일반 대화, 브레인스토밍, 잡담 등은 CHAT으로 분류합니다.\n\n"
        "── **예시: RETRIEVE** ──\n"
        "Q: 우리 회사 연차 규정이 어떻게 되나요?\n"
        "A: RETRIEVE\n"
        "Q: 휴가 종류가 어떻게 되나요?\n"
        "A: RETRIEVE\n"
        "Q: 인사발령 절차 알려줘\n"
        "A: RETRIEVE\n\n"
        "── **예시: USER_INFO** ──\n"
        "Q: 내 남은 연차가 얼마나 있나요?\n"
        "A: USER_INFO\n"
        "Q: 제가 올린 기안 몇 건 있죠?\n"
        "A: USER_INFO\n\n"
        "── **예시: CHAT** ──\n"
        "Q: 오늘 날씨 어때?\n"
        "A: CHAT\n"
        "Q: 점심 뭐 먹을지 추천해줘\n"
        "A: CHAT\n"
        "Q: 주말에 뭐 할까?\n"
        "A: CHAT\n\n"
        "────────────────────\n"
        "질문: {question}\n"
        "답:"
    )
)
router_chain = router_prompt | llm | (lambda x: x.content.strip().upper())

# ——— 스트리밍 엔드포인트 ———
@app.post("/query-stream")
async def query_stream(payload: QueryPayload, request: Request):
    token = request.headers.get("Authorization")  # ✅ 프론트에서 온 토큰 받기
    user_info = await fetch_user_info(token) 
    question = payload.question
    route = router_chain.invoke({"question": question})

    # 사용자 정보 조회
    employee_id = user_info.get("employeeId") 
    if not employee_id:
        return {"error": "employeeId가 없습니다"}

    user_profile = format_user_profile(user_info)


    # 2) 스트리밍 LLM 준비
    llm_stream = ChatOpenAI(
        temperature=0.0,
        model="gpt-4o-mini",
        streaming=True,
        openai_api_key=OPENAI_KEY,
    )

    today   = datetime.now().strftime("%Y년 %m월 %d일")

    if route == "USER_INFO":
    
        # 개인 프로필 질문: 문서 검색 없이 바로 답변
        return StreamingResponse(
            user_info_generator(user_info, question),
            media_type="text/plain"
        )


    elif route == "RETRIEVE":
        # RAG 체인(streaming) — build_rag_chain 내부에서 retriever도 돌아갑니다.
        rag_chain = build_rag_chain(user_info, llm_stream,today)

        async def rag_generator():
            full_resp = ""
            
            # ⚠️ 여긴 dict 형태로 넘겨야 context(search) 변수에 question이 들어갑니다.
            async for chunk in rag_chain.astream(question):
                full_resp += chunk.content
                yield chunk.content
            shared_memory.save_context({"input": question}, {"output": full_resp})

        return StreamingResponse(rag_generator(), media_type="text/plain")

    else:  # CHAT 분기
        async def chat_generator():
            # 1) 히스토리 + 시스템 메시지 구성
            today = datetime.now().strftime("%Y년 %m월 %d일")
            date_msg= SystemMessage(content=f"🗓️ 현재 날짜: {today}")
            history = shared_memory.load_memory_variables({})["history"]
            profile = format_user_profile(user_info)
            system_msg = SystemMessage(content=f"👤 사용자 정보:\n{profile}")
            msgs = [date_msg,system_msg] + history + [HumanMessage(content=question)]

            # 2) 스트리밍 + 메모리 저장
            full = ""
            async for chunk in llm_stream.astream(msgs):
                full += chunk.content
                yield chunk.content
            shared_memory.save_context({"input": question}, {"output": full})
 

        return StreamingResponse(chat_generator(), media_type="text/plain")

    
    

# ——— 동기형 QA 엔드포인트 (선택) ———
class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/qa", response_model=QueryResponse)
async def qa(payload: QueryPayload, request: Request):
    token = request.headers.get("Authorization")
    user_info = await fetch_user_info(token)
    question = payload.question
    route = router_chain.invoke({"question": question})

    if route == "RETRIEVE":
        today     = datetime.now().strftime("%Y년 %m월 %d일")
        # 2) build_rag_chain 호출 (streaming 아닌 동기형이니까 llm 사용)
        rag_chain = build_rag_chain(user_info, llm, today)
        # 3) 질문 문자열만 넘겨서 invoke
        response  = rag_chain.invoke(question)
        shared_memory.save_context({"input": question}, {"output": response.content})

        docs    = retriever.get_relevant_documents(question)
        sources = [d.metadata.get("source", "") for d in docs]
        return QueryResponse(answer=response.content, sources=sources)
    else:
        ans = chat_chain(question)
        return QueryResponse(answer=ans, sources=[])

# ——— 헬스체크 ———
@app.get("/")
def root():
    return {"status": "ok"}
