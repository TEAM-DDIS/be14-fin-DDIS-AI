import requests
import os

# 1회성 S3 > pdf 다운로드

# PDF 저장 경로
SAVE_DIR = "data/raw_policies"
os.makedirs(SAVE_DIR, exist_ok=True)

# [OPTIONAL] 인증이 필요한 경우를 위한 주석 처리 (추후 사용 가능)
# token = "your-jwt-token"
# headers = {"Authorization": f"Bearer {token}"}
headers = {}

def fetch_pdfs():
    print("Spring API로 파일 목록 요청 중...")
    response = requests.get("http://localhost:8000/policies/files", headers=headers)

    if response.status_code != 200:
        print(f"요청 실패: {response.status_code} - {response.text}")
        return

    files = response.json()

    for file in files:
        filename = file["fileName"]
        url = file["downloadUrl"]
        save_path = os.path.join(SAVE_DIR, filename)

        # 이미 다운로드된 파일은 건너뛰기
        if os.path.exists(save_path):
            print(f"이미 있음: {filename}")
            continue

        print(f"⬇다운로드 중: {filename}")
        r = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(r.content)

    print("모든 PDF 다운로드 완료!")

if __name__ == "__main__":
    fetch_pdfs()
