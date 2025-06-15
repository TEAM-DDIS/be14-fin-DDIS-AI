from langchain_community.vectorstores import FAISS

# FAISS 인덱스 불러오기
vectorstore = FAISS.load_local(
    folder_path="data/vectorstore",
    embeddings=None,
    index_name="index",
    allow_dangerous_deserialization=True, 
)

# 내부 문서 보기
docs = vectorstore.docstore._dict.values()

for i, doc in enumerate(list(docs)[:3]):
    print(f"\n문서 #{i+1}")
    print("내용:", doc.page_content[:300], "...")
    print("메타데이터:", doc.metadata)
