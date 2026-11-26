# Python 3.12 슬림 베이스 이미지 사용
FROM python:3.12-slim

# 작업 디렉터리 설정
WORKDIR /app

# 시스템 빌드 의존성 및 curl 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 패키지 목록 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 전체 소스 복사
COPY . .

# Cloud Run의 기본 PORT 환경변수(8080) 대응
ENV PORT=8080
EXPOSE 8080

# 컨테이너 시작 시:
# 1. FastAPI 로그 수집기(Uvicorn)를 백그라운드(8000번 포트)에서 실행
# 2. Streamlit 웹 앱(app2.py)을 Cloud Run 포트(8080)로 실행
ENTRYPOINT ["sh", "-c", "uvicorn logger_api:app --host 127.0.0.1 --port 8000 & exec streamlit run app2.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
