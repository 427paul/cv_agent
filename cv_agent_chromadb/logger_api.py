"""
logger_api.py

FastAPI 기반의 경량 비동기 로그 수집기 API 서비스
- Streamlit(app2.py) 방문자의 버튼 클릭 이벤트 및 LLM 대화 로그 수집
- Google BigQuery 데이터셋(cv_agent) 및 테이블(click_logs, chat_logs)로 실시간 적재
- Cloud Run 환경에서는 서비스 계정으로 자동 무인증 연동
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

# 한국 표준시(KST: UTC+9) 타임존 정의
KST = timezone(timedelta(hours=9))

def get_current_time_info() -> Tuple[str, str]:
    """한국 표준시(KST, UTC+9) 기준의 ISO 타임스탬프(+09:00 포함)와 직관적인 문자열('YYYY-MM-DD HH:MM:SS')을 반환합니다."""
    now_kst = datetime.now(KST)
    return now_kst.isoformat(), now_kst.strftime("%Y-%m-%d %H:%M:%S")

try:
    from fastapi import FastAPI, BackgroundTasks, HTTPException
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None
    BackgroundTasks = None
    HTTPException = None
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_agent_logger")

if HAS_FASTAPI:
    app = FastAPI(
        title="CV Agent Log Collector API",
        description="FastAPI logger to collect user interactions and stream to BigQuery",
        version="1.0.0"
    )
else:
    app = None

# ==============================================================================
# BigQuery 환경 설정 및 초기화
# ==============================================================================
DATASET_ID = os.environ.get("BIGQUERY_DATASET_ID", "cv_agent")
CLICK_TABLE_ID = "click_logs"
CHAT_TABLE_ID = "chat_logs"
EVAL_TABLE_ID = "eval_logs"

bq_client = None
bq_initialized = False


def get_bigquery_client():
    """BigQuery 클라이언트를 싱글톤으로 가져오거나 초기화합니다."""
    global bq_client, bq_initialized
    if bq_client is not None:
        return bq_client

    try:
        from google.cloud import bigquery
        project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        bq_client = bigquery.Client(project=project_id) if project_id else bigquery.Client()
        logger.info(f"BigQuery Client connected successfully for project: {bq_client.project}")

        # 데이터셋 및 테이블 준비
        _ensure_tables_exist(bq_client)
        bq_initialized = True
        return bq_client
    except Exception as e:
        logger.warning(f"BigQuery 연결 초기화 실패 (로컬 모드 또는 인증 없음): {e}")
        return None


def _sync_table_schema(client, table_ref, desired_schema):
    """
    기존 테이블을 보존하면서 새로운 SchemaField(NULLABLE)를 안전하게 추가합니다.
    기존의 어떤 데이터도 삭제되거나 변경되지 않습니다.
    """
    try:
        table = client.get_table(table_ref)
        existing_names = {f.name for f in table.schema}
        new_fields = list(table.schema)
        schema_changed = False

        for field in desired_schema:
            if field.name not in existing_names:
                new_fields.append(field)
                schema_changed = True
                logger.info(f"테이블 '{table_ref.table_id}'에 새 컬럼 '{field.name}'({field.field_type}) 추가 준비")

        if schema_changed:
            table.schema = new_fields
            client.update_table(table, ["schema"])
            logger.info(f"테이블 '{table_ref.table_id}' 스키마가 성공적으로 업데이트되었습니다 (기존 데이터 100% 무손실 유지).")
    except Exception as e:
        logger.warning(f"테이블 '{table_ref.table_id}' 스키마 동기화 중 알림: {e}")


def _ensure_tables_exist(client):
    """cv_agent 데이터셋 및 필요한 테이블을 확인하고 안전하게 스키마를 업데이트합니다."""
    from google.cloud import bigquery

    # 1. 데이터셋 확인 및 생성
    dataset_ref = bigquery.DatasetReference(client.project, DATASET_ID)
    try:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = os.environ.get("BIGQUERY_LOCATION", "asia-northeast3")
        client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset '{DATASET_ID}' is ready.")
    except Exception as e:
        logger.warning(f"데이터셋 생성 확인 중 알림: {e}")

    # 2. click_logs 테이블 스키마 (user_id, kst_time 추가)
    click_table_ref = dataset_ref.table(CLICK_TABLE_ID)
    click_schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("kst_time", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("button_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("page", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("extra_info", "STRING", mode="NULLABLE"),
    ]
    try:
        table = bigquery.Table(click_table_ref, schema=click_schema)
        client.create_table(table, exists_ok=True)
        _sync_table_schema(client, click_table_ref, click_schema)
        logger.info(f"Table '{CLICK_TABLE_ID}' is ready.")
    except Exception as e:
        logger.warning(f"click_logs 테이블 확인 중 알림: {e}")

    # 3. chat_logs 테이블 스키마 (user_id, provider, kst_time 추가)
    chat_table_ref = dataset_ref.table(CHAT_TABLE_ID)
    chat_schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("kst_time", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("user_query", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("llm_response", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("node", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("model_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("provider", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("retrieved_docs_count", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("elapsed_seconds", "FLOAT", mode="NULLABLE"),
    ]
    try:
        table = bigquery.Table(chat_table_ref, schema=chat_schema)
        client.create_table(table, exists_ok=True)
        _sync_table_schema(client, chat_table_ref, chat_schema)
        logger.info(f"Table '{CHAT_TABLE_ID}' is ready.")
    except Exception as e:
        logger.warning(f"chat_logs 테이블 확인 중 알림: {e}")

    # 4. eval_logs 테이블 스키마 (user_id, kst_time 추가)
    eval_table_ref = dataset_ref.table(EVAL_TABLE_ID)
    eval_schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("kst_time", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("eval_content", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_query", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("llm_response", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("target_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
    ]
    try:
        table = bigquery.Table(eval_table_ref, schema=eval_schema)
        client.create_table(table, exists_ok=True)
        _sync_table_schema(client, eval_table_ref, eval_schema)
        logger.info(f"Table '{EVAL_TABLE_ID}' is ready.")
    except Exception as e:
        logger.warning(f"eval_logs 테이블 확인 중 알림: {e}")


# ==============================================================================
# 데이터 모델 정의 (Pydantic)
# ==============================================================================
class ClickLogRequest(BaseModel):
    session_id: Optional[str] = "default_session"
    user_id: Optional[str] = "anonymous"
    button_name: str
    page: Optional[str] = "portfolio_assistant"
    extra_info: Optional[str] = None


class ChatLogRequest(BaseModel):
    session_id: Optional[str] = "default_session"
    user_id: Optional[str] = "anonymous"
    user_query: str
    llm_response: str
    node: Optional[str] = "chat_bot"
    model_name: Optional[str] = "gemini-3.5-flash-lite"
    provider: Optional[str] = "google_ai_studio"
    retrieved_docs_count: Optional[int] = 0
    elapsed_seconds: Optional[float] = None


class EvalLogRequest(BaseModel):
    session_id: Optional[str] = "default_session"
    user_id: Optional[str] = "anonymous"
    eval_content: str
    user_query: str
    llm_response: Optional[str] = ""
    target_type: Optional[str] = "general"
    status: Optional[str] = "NEW"


# ==============================================================================
# BigQuery 스트리밍 적재 함수
# ==============================================================================
def _insert_to_bigquery(table_id: str, row_dict: Dict[str, Any]):
    """Background Task 또는 직접 호출로 실행되는 BigQuery 스트리밍 삽입 로직"""
    client = get_bigquery_client()
    if client is None:
        logger.info(f"[Local Fallback Log] Table: {table_id} | Data: {row_dict}")
        return

    table_full_id = f"{client.project}.{DATASET_ID}.{table_id}"
    try:
        errors = client.insert_rows_json(table_full_id, [row_dict])
        if errors:
            logger.error(f"BigQuery Insert Errors ({table_id}): {errors}")
        else:
            logger.info(f"Successfully streamed 1 row into {table_full_id}")
    except Exception as e:
        logger.error(f"Failed to insert row to BigQuery {table_id}: {e}")


def insert_eval_log_direct(
    user_query: str,
    eval_content: str,
    llm_response: str = "",
    session_id: str = "default_session",
    user_id: str = "anonymous",
    target_type: str = "general",
    status: str = "NEW"
):
    """FastAPI 엔드포인트를 거치지 않고 Python 코드에서 직접 BigQuery eval_logs에 적재하는 헬퍼"""
    iso_ts, kst_str = get_current_time_info()
    row = {
        "timestamp": iso_ts,
        "kst_time": kst_str,
        "session_id": session_id,
        "user_id": user_id,
        "eval_content": eval_content,
        "user_query": user_query,
        "llm_response": llm_response,
        "target_type": target_type,
        "status": status,
    }
    _insert_to_bigquery(EVAL_TABLE_ID, row)


# ==============================================================================
# 엔드포인트 정의
# ==============================================================================
if app is not None:
    @app.get("/health")
    def health_check():
        iso_ts, kst_str = get_current_time_info()
        return {
            "status": "healthy",
            "timestamp": iso_ts,
            "kst_time": kst_str,
            "bigquery_dataset": DATASET_ID,
            "tables": [CLICK_TABLE_ID, CHAT_TABLE_ID, EVAL_TABLE_ID],
        }

    @app.post("/log/click")
    def log_click_event(data: ClickLogRequest, background_tasks: BackgroundTasks):
        """방문자의 버튼 클릭 이벤트 수집 (추천 질문 클릭, 재구축 클릭 등)"""
        iso_ts, kst_str = get_current_time_info()
        row = {
            "timestamp": iso_ts,
            "kst_time": kst_str,
            "session_id": data.session_id,
            "user_id": data.user_id,
            "button_name": data.button_name,
            "page": data.page,
            "extra_info": data.extra_info,
        }
        background_tasks.add_task(_insert_to_bigquery, CLICK_TABLE_ID, row)
        return {"status": "accepted", "event": "click"}

    @app.post("/log/chat")
    def log_chat_event(data: ChatLogRequest, background_tasks: BackgroundTasks):
        """방문자의 LLM 질의응답 내역 수집"""
        iso_ts, kst_str = get_current_time_info()
        row = {
            "timestamp": iso_ts,
            "kst_time": kst_str,
            "session_id": data.session_id,
            "user_id": data.user_id,
            "user_query": data.user_query,
            "llm_response": data.llm_response,
            "node": data.node,
            "model_name": data.model_name,
            "provider": data.provider,
            "retrieved_docs_count": data.retrieved_docs_count,
            "elapsed_seconds": data.elapsed_seconds,
        }
        background_tasks.add_task(_insert_to_bigquery, CHAT_TABLE_ID, row)
        return {"status": "accepted", "event": "chat"}

    @app.post("/log/eval")
    def log_eval_event(data: EvalLogRequest, background_tasks: BackgroundTasks):
        """방문자의 프로젝트 및 Agent 평가/피드백 수집 (BigQuery eval_logs 적재)"""
        iso_ts, kst_str = get_current_time_info()
        row = {
            "timestamp": iso_ts,
            "kst_time": kst_str,
            "session_id": data.session_id,
            "user_id": data.user_id,
            "eval_content": data.eval_content,
            "user_query": data.user_query,
            "llm_response": data.llm_response,
            "target_type": data.target_type or "general",
            "status": data.status or "NEW",
        }
        background_tasks.add_task(_insert_to_bigquery, EVAL_TABLE_ID, row)
        return {"status": "accepted", "event": "eval"}


