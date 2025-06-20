import os
import requests

SPRING_DOWNLOAD_API = "http://localhost:8000/s3/download-url"
SAVE_DIR = "./data/raw_policies"

os.makedirs(SAVE_DIR, exist_ok=True)

# S3에 있는 파일 키
files = {
    "hr_policy.pdf": "company-policy/hr_policy.pdf",
    "salary_regulation.pdf": "company-policy/salary_regulation.pdf",
    "benefit_guide.pdf": "company-policy/benefit_guide.pdf",
    "general_rules.pdf": "company-policy/general_rules.pdf",
    "user_manual.pdf": "company-policy/user_manual.pdf"
}

for filename, s3_key in files.items():
    try:
        # Step 1: presigned GET URL 발급
        res = requests.get(SPRING_DOWNLOAD_API, params={
            "filename": s3_key,
            "contentType": "application/pdf"
        })

        if res.status_code != 200:
            print(f"[X] Failed to get download URL for {filename}: {res.text}")
            continue

        download_url = res.text.strip('"')  # 서버가 문자열만 리턴할 경우

        # Step 2: 파일 다운로드
        pdf_response = requests.get(download_url)
        if pdf_response.status_code == 200:
            with open(os.path.join(SAVE_DIR, filename), 'wb') as f:
                f.write(pdf_response.content)
            print(f"[✓] 다운로드 완료: {filename}")
        else:
            print(f"[X] 다운로드 실패: {filename} ({pdf_response.status_code})")

    except Exception as e:
        print(f"[!] 오류 발생: {filename} → {e}")
