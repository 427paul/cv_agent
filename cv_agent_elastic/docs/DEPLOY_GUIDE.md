# 🚀 Cloud Run & Elasticsearch 기반 AI 포트폴리오 배포 가이드 (`cv_agent_elastic`)

이 문서는 **최규영 AI Portfolio Assistant — Elasticsearch RAG 버전 (`app2.py`)**을 **Elastic Cloud (또는 외부 Elasticsearch 8.x)** 및 **Google Cloud Run**에 배포하는 방법을 안내합니다.

---

## 1. 사전 준비: Elasticsearch 클러스터

Cloud Run은 무상태(Stateless) 컨테이너이므로, 운영 배포 시 외부 Elasticsearch 클러스터를 연결합니다:

- **방법 A (권장): Elastic Cloud / Elastic Serverless**
  1. [Elastic Cloud](https://cloud.elastic.co/)에서 Deployment 생성 후 **Endpoint URL** 또는 **Cloud ID**와 **API Key** 발급
  2. 환경 변수 설정:
     - `ELASTICSEARCH_CLOUD_ID` (또는 `ELASTICSEARCH_URL`)
     - `ELASTICSEARCH_API_KEY`
     - `ELASTICSEARCH_INDEX=portfolio_rag_index`
     - `ELASTICSEARCH_SEARCH_MODE=hybrid` (또는 `bm25`, `vector`)
- **방법 B: 자체 호스팅 Elasticsearch (GCE / Kubernetes 등)**
  - `ELASTICSEARCH_URL=https://<HOST>:9200`
  - `ELASTICSEARCH_USERNAME=elastic`
  - `ELASTICSEARCH_PASSWORD=<PASSWORD>`
  - 자체 서명 인증서 사용 시 `ELASTICSEARCH_VERIFY_CERTS=false`

---

## 2. Google Cloud Run 원클릭 배포

```bash
cd cv_agent_elastic

gcloud run deploy cv-agent-elastic \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --set-env-vars \
GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY",\
OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY",\
ELASTICSEARCH_URL="https://your-es-endpoint:443",\
ELASTICSEARCH_API_KEY="YOUR_ES_API_KEY",\
ELASTICSEARCH_INDEX="portfolio_rag_index",\
ELASTICSEARCH_SEARCH_MODE="hybrid",\
LANGCHAIN_TRACING_V2="true",\
LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY",\
LANGCHAIN_PROJECT="cv-agent-elastic"
```

---

## 3. 검색 모드 선택 가이드

| `ELASTICSEARCH_SEARCH_MODE` | 특징 | 권장 상황 |
| :--- | :--- | :--- |
| `hybrid` (기본값) | BM25 키워드 검색 + kNN 벡터 검색을 RRF로 결합 | 일반 운영 환경 (고유명사 + 의미 검색 모두 최적화) |
| `bm25` | 순수 Elasticsearch BM25 풀텍스트 검색 | 임베딩 API 비용/할당량 절감, OpenRouter 단독 운영 시 |
| `vector` | Dense Vector (`cosine`) kNN 검색 | 순수 의미 기반 유사도 검색 비교·평가 시 |

---

## 4. 배포 삭제

```bash
gcloud run services delete cv-agent-elastic --region asia-northeast3
```
