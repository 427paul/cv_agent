# 🚀 Cloud Run & BigQuery 기반 AI 포트폴리오 배포 가이드

이 패키지는 **최규영 AI Portfolio Assistant (`app2.py`)**와 **FastAPI 기반 실시간 BigQuery 로그 수집기 (`logger_api.py`)**를 Google Cloud Run에 원클릭으로 배포할 수 있도록 구성된 올인원 배포 디렉터리입니다.

---

## 🏗️ 시스템 아키텍처

```text
[방문자 웹 브라우저]
       │
       ▼ (HTTPS :8080)
┌────────────────────────────────────────────────────────┐
│ Google Cloud Run (단일 컨테이너)                          │
│                                                        │
│  ┌──────────────────────┐   내부 비동기 HTTP   ┌───────────────┐  │
│  │ Streamlit (app2.py)  │ ─────────────────▶ │ FastAPI Logger│  │
│  │ (챗봇 UI & RAG 엔진)  │   (127.0.0.1:8000) │(logger_api.py)│  │
│  └──────────────────────┘                    └───────┬───────┘  │
└──────────────────────────────────────────────────────┼─────────┘
                                                       │ 실시간 스트리밍 적재
                                                       ▼
                                         ┌───────────────────────────┐
                                         │ Google BigQuery           │
                                         │ - Dataset: cv_agent       │
                                         │   ├── chat_logs (대화 로그)│
                                         │   ├── click_logs (클릭 로그)│
                                         │   └── eval_logs (평가 로그)│
                                         └───────────────────────────┘
```

---

## 📁 디렉터리 파일 구성

| 파일명 | 용도 및 설명 |
| :--- | :--- |
| `app2.py` | 방문자 API 키 입력창이 제거된 메인 Streamlit 챗봇 (버튼/대화 발생 시 비동기 로깅) |
| `logger_api.py` | FastAPI 기반 비동기 로그 수집기 (BigQuery 테이블 자동 생성 및 스트리밍 적재) |
| `rag_engine.py` | ChromaDB 벡터스토어 및 LangGraph (`rag_bot`, `chat_bot`) 코어 엔진 |
| `PROFILE.md` 등 (4개) | RAG 질의응답에 사용되는 포트폴리오 핵심 마크다운 문서 |
| `Dockerfile` | FastAPI와 Streamlit을 하나의 컨테이너로 패키징하는 도커 빌드 파일 |
| `requirements.txt` | Cloud Run 실행에 필요한 파이썬 패키지 목록 (FastAPI, BigQuery 등 포함) |
| `.dockerignore` | 빌드 시 제외할 캐시 및 불필요한 파일 목록 |

---

## 📊 BigQuery 로그 테이블 스키마

컨테이너가 시작되고 최초 이벤트가 들어오면, BigQuery의 **`cv_agent`** 데이터셋 아래에 3개의 테이블이 **자동 생성**됩니다:

1. **`chat_logs` (대화 기록 테이블)**
   - `timestamp` (TIMESTAMP): 질문 발생 시각 (KST ISO 기준)
   - `kst_time` (STRING): 한국 표준시 읽기 쉬운 문자열 (예: `2026-10-14 00:20:00`)
   - `session_id` (STRING): 브라우저 세션 ID
   - `user_id` (STRING): 방문자 고유 ID (동일 사용자의 다중 질문/세션 추적)
   - `user_query` (STRING): 방문자가 입력한 질문 내용
   - `llm_response` (STRING): LLM 모델이 생성한 답변 내용
   - `node` (STRING): 실행된 노드 (`rag_bot` 또는 `chat_bot`)
   - `model_name` (STRING): 사용된 모델 (`gemini-3.5-flash-lite`, `google/gemini-2.5-flash-lite` 등)
   - `provider` (STRING): 응답 공급자 (`google_ai_studio` 또는 `openrouter`)
   - `retrieved_docs_count` (INTEGER): 검색된 문서 청크 수
   - `elapsed_seconds` (FLOAT): 답변 생성 소요 시간(초)

2. **`click_logs` (클릭 행동 테이블)**
   - `timestamp` (TIMESTAMP): 클릭 발생 시각 (KST ISO 기준)
   - `kst_time` (STRING): 한국 표준시 읽기 쉬운 문자열 (예: `2026-10-14 00:20:00`)
   - `session_id` (STRING): 브라우저 세션 ID
   - `user_id` (STRING): 방문자 고유 ID
   - `button_name` (STRING): 클릭된 버튼 이름 (예: `quick_prompt_profile`, `chromadb_rebuild_button`, `quick_prompt_eval` 등)
   - `page` (STRING): 화면 위치
   - `extra_info` (STRING): 버튼 부가 설명

3. **`eval_logs` (프로젝트 및 Agent 평가/피드백 수집 테이블)**
   - `timestamp` (TIMESTAMP): 평가 작성 일시 (KST ISO 기준)
   - `kst_time` (STRING): 한국 표준시 읽기 쉬운 문자열 (예: `2026-10-14 00:20:00`)
   - `session_id` (STRING): 브라우저 세션 ID
   - `user_id` (STRING): 방문자 고유 ID
   - `eval_content` (STRING): 사용자가 남긴 평가 내용 본문
   - `user_query` (STRING): 전체 입력 원문 (예: `평가: agent에게 프로젝트에 대해서 물어봤는데 설명이 아쉬워요`)
   - `llm_response` (STRING): 일반 채팅봇의 감사 및 접수 응답 내용
   - `target_type` (STRING): 평가 대상 분류 (`agent`, `project`, `general` 등)
   - `status` (STRING): 관리/검토 상태 (`NEW` 기본값, 추후 확인 및 업데이트 가능)


---

## 🚀 배포 방법 (CLI 1단계)

터미널에서 `firebase_deploy` 디렉터리로 이동 후, 아래 명령어 **한 줄**만 실행하면 즉시 배포가 완료됩니다:

```bash
cd firebase_deploy

gcloud run deploy cv-agent-app \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY",OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY",LANGCHAIN_TRACING_V2="true",LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY",LANGCHAIN_PROJECT="cv-agent-portfolio"
```


> [!TIP]
> 배포가 끝나면 화면에 **`Service URL: https://cv-agent-app-xxxx.a.run.app`** 링크가 출력됩니다. 이 링크를 누르면 누구나 즉시 웹사이트를 이용할 수 있습니다.

---

## 🔍 BigQuery 로그 조회 예시 SQL

Google Cloud 콘솔의 BigQuery 스튜디오에서 아래 SQL로 실시간 방문자 로그를 확인할 수 있습니다:

```sql
-- 1. 최근 대화 내역 20건 조회 (한국 시간 kst_time 직접 조회)
SELECT 
  kst_time,
  timestamp,
  user_query,
  llm_response,
  node,
  elapsed_seconds
FROM `<YOUR_GCP_PROJECT_ID>.cv_agent.chat_logs`
ORDER BY timestamp DESC
LIMIT 20;

-- 2. 가장 많이 클릭된 추천 질문 TOP 5 통계
SELECT 
  button_name,
  extra_info,
  COUNT(*) AS click_count
FROM `<YOUR_GCP_PROJECT_ID>.cv_agent.click_logs`
GROUP BY button_name, extra_info
ORDER BY click_count DESC;
```

---

## 🔄 코드 수정 후 재배포 (기존 배포를 내릴 필요 없음 ❌)

코드나 문서를 수정한 후 **기존 배포를 미리 삭제하거나 중단할 필요가 없습니다.**
수정 후 동일한 배포 명령어를 다시 실행하면 Cloud Run이 **무중단(Zero-Downtime)**으로 새 버전으로 자동 교체하며, **기존 접속 URL도 그대로 유지**됩니다:

```bash
cd firebase_deploy

gcloud run deploy cv-agent-app \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY",OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY",LANGCHAIN_TRACING_V2="true",LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY",LANGCHAIN_PROJECT="cv_agent"
```

---

## 🛑 배포된 서비스 완전히 내리기 (삭제)

더 이상 웹 서비스를 운영하지 않고 완전히 삭제하고 싶을 때는 아래 명령어를 실행합니다:

```bash
gcloud run services delete cv-agent-app --region asia-northeast3
```
> 실행 시 삭제 확인 질문(`Do you want to continue (Y/n)?`)이 나오면 `y`를 입력하면 즉시 서비스가 삭제됩니다.

## Google Gemini API Key 등록
```
export GOOGLE_API_KEY="AIzaSy..."
```