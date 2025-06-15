import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings  # 또는 sllm 임베딩으로 교체 가능
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = "data/raw_policies"
VECTOR_DIR = "data/vectorstore"

# 전체 PDF 파일에 대한 태그 사전
TAGS = {
    "personnel_policy_2025-06-15.pdf": ["인사", "근무", "규정"],
    "performance_evaluation_2025-06-15.pdf": ["평가", "고과", "성과"],
    "work_discipline_2025-06-15.pdf": ["근무", "복무", "조직"],
    "leave_policy_2025-06-15.pdf": ["휴가", "연차", "근태"],
    "annual_leave_2025-06-15.pdf": ["연차", "월차", "휴가"],
    "overtime_comp_2025-06-15.pdf": ["근무시간", "수당", "야근"],
    "job_transfer_2025-06-15.pdf": ["이동", "전보", "부서"],
    "promotion_2025-06-15.pdf": ["승진", "고과", "인사"],

    "salary_policy_2025-06-15.pdf": ["급여", "보상"],
    "bonus_policy_2025-06-15.pdf": ["보너스", "성과급"],
    "family_event_2025-06-15.pdf": ["경조금", "복지"],
    "welfare_2025-06-15.pdf": ["복지", "후생"],

    "severance_2025-06-15.pdf": ["퇴직", "퇴직금"],
    "severance_interim_2025-06-15.pdf": ["퇴직금", "중간정산"],
    "discipline_2025-06-15.pdf": ["징계", "처벌", "상벌"],

    "document_mgmt_2025-06-15.pdf": ["문서", "관리", "보안"],
    "handover_2025-06-15.pdf": ["인수인계", "업무", "이관"],

    "software_manual_2025-06-15.pdf": ["프로그램", "매뉴얼", "사용법"]
}

os.makedirs(VECTOR_DIR, exist_ok=True)

def load_pdfs():
    all_docs = []

    for file in os.listdir(PDF_DIR):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(PDF_DIR, file)
        print(f"PDF 로딩 중: {file}")
        loader = PyPDFLoader(path)
        docs = loader.load_and_split()

        for doc in docs:
            doc.metadata["fileName"] = file
            doc.metadata["tags"] = TAGS.get(file, []) 
        all_docs.extend(docs)

    return all_docs

def embed_documents(docs):
    print("임베딩 생성 중...")
    embeddings = OpenAIEmbeddings()  # sllm 쓸 경우 여기 교체
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_DIR)
    print("벡터 저장 완료:", VECTOR_DIR)

if __name__ == "__main__":
    docs = load_pdfs()
    embed_documents(docs)
