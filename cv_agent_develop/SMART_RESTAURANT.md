# 자연어 기반 스마트 맛집 추천 플랫폼 (Smart Restaurant Platform)

카카오맵 데이터를 기반으로 비정형 동적 웹 요소와 Canvas 그래픽을 수집·수치화하고, **S3 → Redshift → dbt → ML(K-Means) → Airflow**로 연결하여 사용자의 식당 탐색 및 의사결정을 지원하는 **End-to-End 데이터 엔지니어링 및 추천 플랫폼** 프로젝트입니다.

---

## 📌 1. 프로젝트 개요 (Overview)

- **프로젝트명**: 자연어 기반 스마트 맛집 추천 플랫폼
- **개발 기간**: 2025.12 – 2026.01 (약 1개월)
- **수행 형태**: 4인 팀 프로젝트 (Team5-CatchData)
- **담당 역할**: Data Engineer (데이터 수집, 비정형 이미지 수치화, DW/Data Mart 모델링, ML 클러스터링, 파이프라인 오케스트레이션 및 무결성 모니터링 전담)
- **팀 레포지토리**: [Team5-CatchData GitHub](https://github.com/Team5-CatchData)
- **활용 데이터**: Kakao 지도 API 및 카카오맵 맛집 상세 페이지 (기본 정보, 메뉴, 가격, 영업시간, 평점, 리뷰 및 방문 추이 Canvas 그래프)
- **프로젝트 목표**:
  - 사용자의 자연어 입력을 통해 LLM으로부터 식당을 추천받거나 대시보드를 통해 사용자의 의사결정을 지원

---

## 🛠 2. 기술 스택 (Tech Stack)

| 영역 | 기술 스택 | 설명 |
| :--- | :--- | :--- |
| **Languages & Scraping** | `Python`<br>`SQL`<br>`Selenium`<br>`BeautifulSoup` | 핵심 파이프라인 개발 및 데이터베이스 쿼리 언어<br>동적 웹 렌더링 파싱 및 멀티스레드(`ThreadPoolExecutor`) 병렬 크롤러 |
| **Infrastructure & DW** | `AWS S3`<br>`Amazon Redshift` | 대규모 Raw 데이터 아카이빙용 Data Lake 스토리지<br>분석 및 고속 조회를 위한 클라우드 데이터 웨어하우스(DW) |
| **Data Toolset & Ops** | `Apache Airflow`<br>`dbt (data build tool)`<br>`Docker`<br>`Slack Webhook` | ETL/ELT 및 ML 추론 파이프라인 오케스트레이션 (DAG)<br>Redshift 기반 분석용 Data Mart 비정규화 및 변환<br>일관된 개발/운영 컨테이너 환경 격리<br>파이프라인 실행 상태 및 이상치 장애 실시간 알림 |
| **ML / CV & Analytics** | `OpenCV`<br>`K-Means`<br>`StandardScaler`<br>`Data Mart` | Canvas 렌더링 그래프 픽셀 컬러 마스킹 및 혼잡도 정량화<br>최적 군집(k=8) 기반 식당 테마 분류 및 추천 스코어링<br>다차원 피처 정규화 및 분석용 마트 테이블 구축 |


---

## 🔄 3. 시스템 아키텍처 (Architecture) 및 엔드투엔드 데이터 생애주기 (Data Lifecycle 5단계)

```
[Step 01] 멀티스레드 크롤링 ──▶ [Step 02] Canvas 혼잡도 수치화 ──▶ [Step 03] S3 & Redshift 적재 ──▶ [Step 04] K-Means 클러스터링 ──▶ [Step 05] Airflow & Slack 모니터링
  (ThreadPoolExecutor 1.5배)      (OpenCV Color Masking)           (dbt Data Mart 모델링)         (k=8 최적 군집 분류)            (데이터 무결성 검증)
```

1. **Step 01 · 수집 (Collection)**: 카카오맵 맛집 상세 정보 수집 및 `ThreadPoolExecutor` 멀티스레드 병렬화 최적화
2. **Step 02 · 비정형 피처화 (CV Feature)**: 텍스트 추출이 불가능한 Canvas 방문자 추이 그래프를 OpenCV 컬러 마스킹으로 0~100 혼잡도 수치로 정량화
3. **Step 03 · 웨어하우스 적재 (Warehouse)**: AWS S3에 원천 데이터 보존 후 Amazon Redshift 정규 적재, dbt 기반 분석 마트 모델링
4. **Step 04 · 머신러닝 연동 (ML Modeling)**: StandardScaler 정규화, Elbow & Silhouette 기법을 통한 최적 군집 $k=8$ 도출 및 추천 스코어링 연계
5. **Step 05 · 오케스트레이션 & 모니터링 (Monitoring)**: 전체 파이프라인의 Airflow DAG 표준화, Docker 컨테이너 격리, Slack Webhook 실시간 장애 감지

---

## 🔍 4. 단계별 핵심 구현 상세 (Implementation Breakdown)

### STAGE 01. 대규모 맛집 데이터 수집 및 멀티스레드 병렬화
- **동적 렌더링 파싱**: Selenium과 BeautifulSoup을 결합하여 매장명, 업종, 주소, 메뉴, 가격, 영업시간, 평점, 유저 리뷰 등 세부 항목 파싱
- **병렬 크롤링 아키텍처**: 단일 스레드 순차 크롤링 시 발생하는 네트워크 I/O 병목을 해결하기 위해 Python `ThreadPoolExecutor`를 활용한 동시성 크롤러 설계
- **성능 향상**: 200건 기준 수집 소요 시간을 **8분 30초에서 5분 30초로 약 1.5배 단축**

### STAGE 02. 비정형 Canvas 이미지의 컴퓨터 비전 수치화 (OpenCV)
- **문제 인식**: 식당 추천의 핵심 지표인 '실시간 웨이팅/혼잡도'가 백엔드 API가 아닌 프론트엔드 Canvas 객체로 렌더링되어 일반적인 HTML 파싱 불가
- **Color Masking 기법**:
  - 요일·시간대별 방문자 추이 Canvas 영역을 스크린샷 캡처
  - 특정 RGB/HSV 색상 임계치를 적용하여 배경 격자선을 제거하고 방문자 추이 그래프 선만 마스킹 추출
  - Y축 픽셀 높이의 상대적 위치를 계산하여 0~100 사이의 **'혼잡도 지수(Congestion Index)'**로 변환
- **데이터 자산화**: 비정형 그래픽 요소를 추천 알고리즘의 정량적 수치 피처로 성공적 디지털 전환

### STAGE 03. AWS S3 Data Lake 및 Amazon Redshift DW 모델링
- **Data Lake (S3)**: 수집된 Raw JSON 및 전처리 데이터를 S3 버킷에 파티셔닝 적재하여 데이터 원본 보존 및 유실 방지
- **Data Warehouse (Redshift)**: 대규모 분석 쿼리 최적화를 위한 테이블 스키마 설계 및 배치 적재
- **Data Mart (dbt)**: Redshift 상에서 dbt(data build tool)를 활용해 비정규화 분석 마트를 모델링하여 다차원 분석 및 추천 서빙 지연 최소화

### STAGE 04. K-Means 클러스터링 & 다면 추천 피처 엔지니어링
- **피처 전처리**: OpenCV 혼잡도 지수, 리뷰 감성 분석 점수, 메뉴 평균 가격대, 방문 목적 태그 등 다차원 이종 변수를 `StandardScaler`로 표준화
- **군집 최적화**:
  - 군집 내 오차제곱합(WCSS)을 평가하는 **Elbow Method**와 군집 간 분리도를 측정하는 **Silhouette Score** 복합 평가
  - 데이터 응집도와 분별력이 가장 우수한 **최적 군집 수 $k = 8$** 도출
- **테마 매핑**: 8개 군집을 비즈니스 관점에서 해석하여 테마별(가성비 맛집, 데이트 코스, 단체 회식, 웨이팅 핫플레이스 등) 세분화 매핑 및 가중 추천 스코어링 알고리즘 구축

### STAGE 05. Apache Airflow 파이프라인 오케스트레이션
- **DAG 표준화**: 데이터 수집 → OpenCV CV 수치화 → S3 아카이빙 → Redshift 적재 → dbt 마트 변환 → ML 추론 전 과정을 유기적인 Airflow Task로 모듈화
- **Docker 기반 환경 격리**: 모든 실행 환경을 Docker 컨테이너로 패키징하여 로컬 개발과 서버 환경 간 종속성 문제 제거

### STAGE 06. 데이터 무결성 검증 & Slack Webhook 모니터링
- **데이터 품질 검증 (Validation)**: 각 파이프라인 단계마다 레코드 수집 건수, Null 및 결측치 비율, 비정상적 스키마 변경을 감지하는 유효성 검사 로직 배치
- **Slack Alert**: 실행 완료 보고, 레코드 처리 통계, 이상치 감지 및 실패 시 트레이스백을 Slack Webhook으로 즉각 전송하여 파이프라인 가시성 및 운영 안정성 확보

---

## ⚡ 5. 트러블슈팅 및 문제 해결 (Troubleshooting)

### 🚨 Issue 01: 실시간 웨이팅 및 혼잡도 데이터 부재
- **문제**: 식당 추천 시 가장 중요한 기준 중 하나인 실시간 웨이팅/혼잡도를 직접 나타내는 공개 API나 수치 데이터가 존재하지 않음
- **해결**: 카카오맵에서 제공하는 '요일·시간대별 평균 방문자 데이터'를 대체 지표로 채택하여 시간대별 상대 혼잡도를 추정할 수 있는 정량 피처로 가공

### 🚨 Issue 02: Canvas 그래픽 렌더링으로 인한 텍스트 파싱 불가
- **문제**: 방문자 추이 데이터가 텍스트나 테이블이 아닌 브라우저 Canvas 그래픽으로 렌더링되어 일반적인 Selenium/BeautifulSoup DOM 파싱으로 추출 불가
- **해결**: 해당 요소를 이미지로 캡처한 뒤 OpenCV Color Masking을 적용하여 그래프 곡선 픽셀 좌표를 인식하고, 이를 0~100 스케일의 '혼잡도 지수'로 수치화하여 문제 해결

### 🚨 Issue 03: 동일 Place ID 반복 수집으로 인한 데이터 뻥튀기 (200건 → 1,600건)
- **문제**: 동적 웹 렌더링과 페이징 처리 중복으로 초기 200건의 수집 대상 식당이 가공 파이프라인을 거치며 1,600건 이상으로 비정상 누적되는 정합성 결함 발생
- **원인**: 검색 결과 리스트에서 동일한 식당 고유 식별자(Place ID)가 반복 호출됨
- **해결**:
  - 수집 프로세스 메모리 내에 **Place ID 기반 Set 중복 검사 로직**을 추가하여 1차 수집 단계에서 중복 차단
  - Redshift 적재 전처리 파이프라인에 중복 제거 쿼리(`ROW_NUMBER() OVER (PARTITION BY place_id)`) 및 유효성 검증 경보를 보강하여 완벽한 데이터 무결성 달성

### 🚨 Issue 04: 순차 크롤링의 I/O 병목으로 인한 수집 지연 (8분 30초)
- **문제**: 초기 순차적 동기 크롤링 방식은 페이지 렌더링 대기 및 네트워크 지연으로 200건 기준 8분 30초가 소요되어 대량 수집 시 확장성 한계 직면
- **해결**: Python `concurrent.futures.ThreadPoolExecutor`를 도입하여 워커 기반 병렬 크롤링 구조로 전환, **수집 시간을 5분 30초로 약 1.5배 단축**

---

## 📊 6. 주요 성과 및 정량 지표 (Key Metrics)

| 구분 | AS-IS (개선 전) | TO-BE (개선 후) | 개선 효과 |
| :--- | :---: | :---: | :--- |
| **수집 처리 속도 (200건 기준)** | 8분 30초 | **5분 30초** | ⚡ **1.5배 (35%) 속도 단축** (`ThreadPoolExecutor` 병렬화) |
| **데이터 정합성 (중복 누적)** | 200건 → 1,600건 비정상 누적 | **정확한 1:1 정합성 보장** | 🎯 **Place ID Set 검증 & 전처리 쿼리로 무결성 확보** |
| **비정형 그래픽 자산화** | 추출 불가 (수치 데이터 부재) | **0~100 혼잡도 지수 정량화** | 💡 **OpenCV 컬러 마스킹으로 추천 피처 자산화** |
| **파이프라인 안정성** | 수동 확인 및 장애 미인지 | **실시간 Slack 자동 알림** | 🔔 **이상치 조기 감지 및 파이프라인 가시성 확보** |
