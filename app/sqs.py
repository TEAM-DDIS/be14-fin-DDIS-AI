import os
import json
import time
import tempfile

import boto3
import botocore
from botocore.config import Config

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

# — AWS 클라이언트 초기화 —
# aws_cfg = Config(retries={'max_attempts': 3, 'mode': 'standard'})
aws_cfg = Config(region_name="ap-northeast-2", retries={"max_attempts": 2})
sqs = boto3.client("sqs", config=aws_cfg)
s3  = boto3.client("s3", config=aws_cfg)

# sqs = boto3.client("sqs", region_name="ap-northeast-2")

# SQS 큐 이름 및 URL
QUEUE_NAME = "DDISQueue"
# QUEUE_URL  = sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
QUEUE_URL = "https://sqs.ap-northeast-2.amazonaws.com/395850919569/DDISQueue"

# Chroma 인덱스 저장 경로 (EBS나 영구 스토리지 마운트)
PERSIST_DIR = "/mnt/vector-db/chroma_index"

def process_pdf(bucket: str, key: str):
    """S3 → PDF 다운로드 → PyPDFLoader 로드 → OpenAI 임베딩 → Chroma 저장"""
    print(f"▶ 처리 시작: s3://{bucket}/{key}")
    # 1) S3에서 PDF 다운로드
    try:
        # HEAD 먼저 해 보고
        s3.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        # 404 일 때만 무시
        if e.response['Error']['Code'] == '404':
            print(f"⚠️ 스킵: 객체 없음 {key}")
            return
        else:
            raise
    
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        s3.download_file(bucket, key, tmp.name)

        # 2) PyPDFLoader로 PDF 로드 & 페이지 분할
        loader = PyPDFLoader(tmp.name)
        docs = loader.load_and_split()

    # 3) OpenAI 임베딩 생성
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 4) Chroma에 문서 + 임베딩 저장
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()

    print(f"✅ 처리 완료: {key}")

def poll_and_process():
    """SQS를 롱폴링하며 메시지 수신 → PDF 처리 → 메시지 삭제"""
    print("▶ AI 서버: SQS 폴링을 시작합니다.")
    while True:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=5,
            WaitTimeSeconds=20,    # 롱폴링
            VisibilityTimeout=120  # 처리 시간 여유
        )
        messages = resp.get("Messages", [])
        if not messages:
            continue

        for msg in messages:
            try:
                body = json.loads(msg["Body"])
                for record in body.get("Records", []):
                    bucket = record["s3"]["bucket"]["name"]
                    key    = record["s3"]["object"]["key"]
                    process_pdf(bucket, key)

                # 처리 완료 후 메시지 삭제
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"]
                )
            except Exception as e:
                print(f"⚠️ 메시지 처리 중 오류: {e}")
        time.sleep(1)

if __name__ == "__main__":
    poll_and_process()
