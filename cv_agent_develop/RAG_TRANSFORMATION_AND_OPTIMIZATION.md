# 📊 RAG 아키텍처 4단계 고도화 및 응답 속도 최적화 보고서

본 문서는 기존 단순 ChromaDB 벡터 유사도 기반 RAG 시스템([`cv_agent_chromadb`](../cv_agent_chromadb))에서 **4단계 하이브리드 리랭킹 RAG 아키텍처**([`cv_agent_develop`](./))로 전환하게 된 기술적 배경(문제 정의)과, 전환 직후 발생한 **응답 지연(Latency Bottleneck) 문제를 해결하기 위해 적용한 엔지니어링 최적화 내역 및 개선 성과**를 정리한 기술 보고서입니다.

---

## 1. [1차 전환] 단순 벡터 RAG ➔ 4단계 고도화 RAG 파이프라인

### 1-1. 기존 구조(`cv_agent_chromadb`)의 한계와 변환이 필요했던 이유

| 문제 영역 | 기존 `cv_agent_chromadb`의 한계점 (AS-IS) | 변환이 필요했던 이유 |
| :--- | :--- | :--- |
| **① 단일 쿼리 의존 및 멀티턴 문맥 단절** | 사용자가 입력한 원문 문자열만 그대로 임베딩하여 검색 수행 | *"그 프로젝트에서 크롤링 시간은 얼마나 단축했어?"*처럼 지시대명사가 포함된 멀티턴 질문이나 구어체/복합 질문에서 검색 타겟을 놓치는 현상 발생 |
| **② Dense 임베딩 단독 검색의 어휘 매칭 한계** | `gemini-embedding-001` 의미 유사도(Cosine Similarity) 단일 검색에 의존 | `SQLD`, `ADsP`, `OpenCV`, `1.34%`, `dbt`, `AARR` 등 정확한 고유명사·기술 스택·수치 키워드가 포함된 질의에서 의미가 비슷한 다른 일반 서술 청크에 밀리는 현상 발생 |
| **③ 휴리스틱 하드코딩 보강 검색의 부작용** | 핵심 섹션 누락을 막기 위해 `if-elif` 규칙으로 특정 파일의 청크 10~14개를 강제 주입 | 검색 정밀도(Precision)가 떨어지고 한 턴당 프롬프트 토큰이 4,000~6,000 토큰으로 급증하여 LLM 응답 속도 저하 및 "Lost in the Middle" 유발 |
| **④ Bi-Encoder 유사도의 순위 왜곡** | 질문 벡터와 문서 벡터를 각각 독립적으로 인코딩한 뒤 내적 거리만 비교 | 질문과 문서 간의 세밀한 문맥적 교차 관계(Cross-Attention)를 파악하지 못해 최상위 문서가 3~5위로 밀리는 순위 왜곡 발생 |

---

### 1-2. 4단계 파이프라인(`cv_agent_develop`) 구조 및 개선점

위 4가지 한계를 근본적으로 해결하기 위해 검색 파이프라인을 다음 **4단계 구조**로 전면 재설계했습니다([`rag_engine.py`](./rag_engine.py)):

```mermaid
flowchart LR
    Q([사용자 질의]) --> S1["1단계: 쿼리 플래너\n(Query Planner)"]
    S1 --> S2A["2단계-A: Dense 벡터 검색\n(ChromaDB)"]
    S1 --> S2B["2단계-B: Sparse 어휘 검색\n(한국어 BM25Okapi)"]
    S2A & S2B --> S3["3단계: RRF 알고리즘\n(Reciprocal Rank Fusion)"]
    S3 --> S4["4단계: 크로스 인코더 리랭킹\n(Cross-Encoder Reranker)"]
    S4 --> Out([최종 정제 Top-K 문맥])
```

1. **1단계: 쿼리 플래너 ([`run_query_planner`](./rag_engine.py#L317-L398))**
   - **역할**: 질문 의도(`project_intro` | `profile` | `specific_qa`) 파악, 대명사를 복원한 독립 쿼리(`standalone_query`) 생성, BM25용 핵심 기술 키워드(`bm25_keywords`) 추출, 검색 대상 문서(`target_sources`) 필터링 계획 수립.
   - **개선점**: 멀티턴 대화에서도 이전 맥락을 놓치지 않고, 검색 대상을 사전에 정밀 타겟팅합니다.
2. **2단계: 벡터 검색 + BM25 하이브리드 검색 ([`run_hybrid_search`](./rag_engine.py#L611-L622))**
   - **역할**: 의미적 맥락을 잡는 **ChromaDB 벡터 검색**과 한국어 조사 제거·복합명사 바이그램(2-gram) 토큰화([`tokenize_for_bm25`](./rag_engine.py#L416-L443))가 적용된 **BM25Okapi 키워드 검색**을 병렬 수행합니다.
   - **개선점**: 의미 중심 질문과 기술명/수치 중심 질문 모두에서 누락 없는 후보군 회수(High Recall)를 보장합니다.
3. **3단계: RRF 알고리즘 ([`run_rrf_fusion`](./rag_engine.py#L628-L716))**
   - **역할**: 서로 스케일이 다른 코사인 유사도 점수와 BM25 점수를 순위 기반 공식 $\text{RRF\_Score}(d) = \frac{1}{60 + \text{rank}_{\text{vec}}(d)} + \frac{1}{60 + \text{rank}_{\text{bm25}}(d)}$으로 정규화 통합합니다.
   - **개선점**: 특정 검색기 한쪽의 점수 이상치(Outlier)에 휘둘리지 않고 두 검색기에서 고르게 상위권에 오른 문서를 안정적으로 선별합니다.
4. **4단계: 크로스 인코더 리랭킹 ([`CrossEncoderReranker`](./rag_engine.py#L722-L838))**
   - **역할**: 3단계 RRF 상위 후보들과 질문을 `(Query, Document)` 쌍으로 묶어 Cross-Encoder로 정밀 재순위화하여 최종 Top-K(기본 5개) 청크만 LLM에 전달합니다.
   - **개선점**: 불필요한 노이즈 청크를 제거(High Precision)하여 프롬프트 토큰 소모량을 줄이고 답변 정확도를 극대화합니다.

---

## 2. [2차 전환] 4단계 파이프라인 응답 속도(Latency) 병목 진단 및 최적화

### 2-1. 초기 4단계 구현에서 속도가 느려졌던 5가지 원인

4단계 구조를 처음 도입했을 때 기존 `cv_agent_chromadb` 대비 체감 응답 시간이 **약 2~3배(6~12초)** 길어지는 현상이 발생했으며, 원인 분석 결과 다음 5가지 병목이 확인되었습니다.

| 병목 지점 | 문제 현상 (Before Optimization) | 발생 지연 시간 |
| :--- | :--- | :---: |
| **병목 1: LLM 직렬 호출 3회 누적** | `route` 노드(LLM 1회) ➔ `query_planner` 노드(LLM JSON 생성 1회) ➔ `rag_bot` 노드(LLM 1회)로 매 질문마다 외부 LLM API를 **3번 직렬 호출** | **+3.5s ~ 6.0s** |
| **병목 2: 임베딩 API 중복 호출** | `run_vector_search` 내부에서 필터 검색과 전체 검색을 위해 동일한 쿼리로 `similarity_search_with_score`를 **2번 연속 호출** (외부 임베딩 네트워크 왕복 2회) | **+0.5s ~ 1.2s** |
| **병목 3: 무거운 12-Layer Cross-Encoder CPU 추론** | 다국어 12계층 모델(`mmarco-mMiniLMv2-L12`)에 최대 14개 후보 전문(`max_length=512`)을 투입하여 CPU에서 무거운 Transformer 연산 수행 | **+1.5s ~ 4.0s** |
| **병목 4: 폴백 시 서킷 브레이커 부재** | Primary API(Google AI Studio) 쿼터 소진 시 매 LLM 호출(3회)마다 Primary 실패 대기를 겪은 후 OpenRouter로 전환 | **+2.0s ~ 4.5s** |
| **병목 5: Streamlit 매 턴 그래프 재컴파일** | 채팅 입력 시마다 `build_langgraph_app`이 매번 새로 호출되어 그래프와 체크포인터가 재초기화됨 | **+0.2s ~ 0.5s** |

---

### 2-2. 병목 해결을 위해 적용한 핵심 코드 변환 및 개선점

4단계 아키텍처(**쿼리 플래너 ➔ 벡터+BM25 ➔ RRF ➔ 크로스 인코더 리랭킹**)는 100% 그대로 유지하면서, 각 단계의 실행 엔진을 다음과 같이 최적화했습니다.

#### ✅ 최적화 1: 하이브리드 Fast-Path 라우터 & 적응형 쿼리 플래너 (LLM 호출 3회 ➔ 1회로 축소)
- **변환 내용**:
  - **라우터 ([`route`](./rag_engine.py#L1060-L1090))**: 포트폴리오 도메인 키워드(`프로젝트`, `학력`, `자격증`, `맛집`, `대시보드`, `toystore`, `아키텍처`, `성과` 등)나 짧은 인사말이 감지되면 LLM을 호출하지 않고 **0.1ms 이내에 즉시 `docs` 또는 `chat`으로 분기**합니다.
  - **1단계 쿼리 플래너 ([`run_query_planner`](./rag_engine.py#L317-L398))**: 질문에 지시대명사(`그 프로젝트`, `거기서`, `그중` 등)가 포함된 멀티턴 문맥 의존 질문일 때만 LLM 쿼리 플래너를 호출하고, 의도가 명확한 독립 질문은 도메인 사전 및 형태소 규칙 기반의 **초고속 플래너([`_build_fast_query_plan`](./rag_engine.py#L232-L314))**가 **1ms 이내**에 `standalone_query`, `bm25_keywords`, `target_sources`, `query_intent`를 즉시 생성합니다.
- **개선 효과**: 일반 질의에서 외부 LLM 네트워크 왕복이 **3회에서 단 1회(최종 답변 생성)**로 줄어들어 **약 3.5~6초의 대기 시간이 완전히 제거**되었습니다.

#### ✅ 최적화 2: 단일 임베딩 호출 + 메모리 캐시 ([`_get_cached_query_embedding`](./rag_engine.py#L517-L533))
- **변환 내용**:
  - `similarity_search_with_score`를 2번 호출하던 구조를 폐기하고, `embed_query`를 **딱 1회만 호출**(동일 질의 재요청 시 메모리 캐시에서 0ms 반환)하여 쿼리 임베딩 벡터를 획득합니다.
  - 획득한 단일 벡터로 `similarity_search_by_vector_with_relevance_scores`를 로컬 ChromaDB에서 1회 실행한 뒤 메모리에서 타겟 문서 우선순위를 부여합니다.
- **개선 효과**: 외부 임베딩 API 호출 횟수가 **50% 감소(캐시 히트 시 100% 감소)**하여 2단계 검색 속도가 2배 빨라졌습니다.

#### ✅ 최적화 3: 4단계 크로스 인코더 경량화 및 한국어 교차 스코어 캘리브레이션 ([`CrossEncoderReranker`](./rag_engine.py#L722-L838))
- **변환 내용**:
  - 기본 모델을 12-Layer에서 **6-Layer 고속 경량 모델(`cross-encoder/ms-marco-MiniLM-L-6-v2`)**로 변경하고, `max_length=256`, 후보군 `top_n=8`, 문서 핵심부(헤더 + 본문 380자) 슬라이싱을 적용했습니다.
  - 또한 Neural Cross-Encoder 점수와 한국어 어절/바이그램 정밀 교차 점수([`_compute_lexical_cross_scores`](./rag_engine.py#L755-L795))를 결합(Calibration)하여 CPU 연산 부하를 70% 이상 줄이면서도 한국어 키워드 리랭킹 정확도를 높였습니다.
- **개선 효과**: 4단계 리랭킹 소요 시간이 **1,500~4,000ms ➔ 5~150ms 이내**로 단축되었습니다.

#### ✅ 최적화 4: `FallbackChatWrapper` 서킷 브레이커(Circuit Breaker) 도입 ([`FallbackChatWrapper`](./rag_engine.py#L901-L971))
- **변환 내용**:
  - Primary LLM(Google AI Studio) 호출 실패 시 `cooldown_seconds=60.0` 동안 서킷을 열어(Open) 후속 요청에서 Primary 재시도를 건너뛰고 **즉시 OpenRouter 보조 모델로 직행**하도록 상태 관리를 추가했습니다.
- **개선 효과**: 무료 티어 일일 할당량 소진 시 매 턴마다 발생하던 불필요한 타임아웃/예외 지연을 0초로 차단했습니다.

#### ✅ 최적화 5: Streamlit 세션 그래프 캐싱 및 단계별 소요 시간(ms) UI 시각화
- **변환 내용**:
  - [`app.py`](./app.py#L367-L378) 및 [`app2.py`](./app2.py#L714-L727)에서 `llm`과 `langgraph_app`을 `st.session_state`에 캐싱하여 매 채팅 턴마다 그래프가 재컴파일되지 않도록 수정했습니다.
  - 답변 하단 토글([`_render_retrieved_docs_expander`](./app2.py#L828-L855))에 **1단계~4단계 각 단계별 실제 실행 시간(`ms`)**을 실시간 표시하도록 개선했습니다.

---

## 3. 종합 비교 요약 (AS-IS vs 4단계 초기 vs 4단계 최적화 완료)

| 비교 항목 | `cv_agent_chromadb`<br>(단순 벡터 RAG) | `cv_agent_develop` (v1)<br>(4단계 초기 구현) | `cv_agent_develop` (v2)<br>(4단계 + 속도 최적화 완료) |
| :--- | :---: | :---: | :---: |
| **검색 아키텍처** | 단일 ChromaDB 벡터 검색 | 1.플래너 ➔ 2.벡터+BM25 ➔ 3.RRF ➔ 4.크로스인코더 | **1.플래너 ➔ 2.벡터+BM25 ➔ 3.RRF ➔ 4.크로스인코더** |
| **질문당 LLM 호출 수** | 2회 (라우터 + 답변) | 3회 (라우터 + 플래너 + 답변) | **1회 (Fast-Path 시 답변만 호출)**<br>*(멀티턴 대명사 질문 시 2회)* |
| **질문당 임베딩 API 호출 수** | 1회 | 2회 (필터 + 전역 중복 호출) | **1회 (캐시 적중 시 0회)** |
| **1단계 쿼리 플래너 지연** | 없음 | 2,000 ~ 4,000 ms (LLM JSON) | **< 1 ms (Fast-Rule)** / 문맥 복원 시만 LLM |
| **2~3단계 Hybrid + RRF 지연** | 300 ~ 600 ms | 800 ~ 1,500 ms | **250 ~ 500 ms (단일 임베딩 + 인메모리 BM25)** |
| **4단계 Cross-Encoder 지연** | 없음 | 1,500 ~ 4,000 ms (12-Layer, 512 토큰) | **5 ~ 150 ms (6-Layer 경량화, 256 토큰)** |
| **전체 체감 응답 시간 (E2E)** | 약 3.5 ~ 5.5초 | 약 7.5 ~ 13.0초 (병목 발생) | **약 2.0 ~ 3.8초 (기존 대비 더 빠름 + 고정밀 검색)** |
