"""
app2.py

Streamlit 기반의 최규영 포트폴리오 AI 어시스턴트 웹 애플리케이션 (Elasticsearch RAG + 고정 API 키 & 배포용 버전)
- 사용자에게 API 키를 요구하지 않고 시스템에 설정된 Google AI Studio / OpenRouter API 키 및 Elasticsearch 설정으로 작동
- 4개 문서(PROFILE.md, RAG_DASHBOARD.md, SMART_RESTAURANT.md, TOYSTORE_MOBILE_REVENUE.md)를
  Elasticsearch에 적재하고 Hybrid(BM25 + kNN RRF) / BM25 / Vector(kNN) 검색 기반 RAG 응답 생성
- LangGraph를 활용한 문서 기반 Q&A (rag_bot) 및 일반 대화 (chat_bot) 라우팅
- FastAPI & BigQuery 비동기 로그 수집 + LangSmith 트레이싱 지원
"""

import os
import time
import uuid
import streamlit as st
from rag_engine import (
    DEFAULT_ES_INDEX,
    DEFAULT_ES_URL,
    DEFAULT_SEARCH_MODE,
    DOC_METADATA_CONFIG,
    QUOTA_EXHAUSTED_MESSAGE,
    ElasticsearchConfig,
    build_langgraph_app,
    extract_evaluation_info,
    get_or_create_elasticsearch_store,
    init_llm_with_fallback,
    is_quota_exceeded_error,
    run_query,
)

# ==============================================================================
# 0. 고정 API 키 및 Elasticsearch 기본 설정 (Cloud Run / 로컬 배포용)
# ==============================================================================
DEFAULT_GOOGLE_API_KEY = ""
DEFAULT_OPENROUTER_API_KEY = ""
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash-lite"

DEFAULT_LANGCHAIN_API_KEY = ""
DEFAULT_LANGCHAIN_PROJECT = "cv-agent-elastic"


def _resolve_setting(key: str, default: str = "") -> str:
    """st.secrets -> os.environ -> default 순으로 설정값을 해석합니다."""
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    env_val = os.environ.get(key, "").strip()
    if env_val:
        return env_val

    return default.strip()


def resolve_api_key() -> str:
    return _resolve_setting("GOOGLE_API_KEY", DEFAULT_GOOGLE_API_KEY)


def resolve_openrouter_api_key() -> str:
    return _resolve_setting("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_API_KEY)


def resolve_openrouter_model() -> str:
    return _resolve_setting("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL) or DEFAULT_OPENROUTER_MODEL


def resolve_elasticsearch_config(search_mode_override: str = "") -> ElasticsearchConfig:
    """Secrets / 환경변수 기반으로 ElasticsearchConfig를 구성합니다."""
    url = _resolve_setting("ELASTICSEARCH_URL", DEFAULT_ES_URL) or DEFAULT_ES_URL
    index_name = _resolve_setting("ELASTICSEARCH_INDEX", DEFAULT_ES_INDEX) or DEFAULT_ES_INDEX
    cloud_id = _resolve_setting("ELASTICSEARCH_CLOUD_ID", "")
    api_key = _resolve_setting("ELASTICSEARCH_API_KEY", "")
    username = _resolve_setting("ELASTICSEARCH_USERNAME", "")
    password = _resolve_setting("ELASTICSEARCH_PASSWORD", "")
    verify_raw = _resolve_setting("ELASTICSEARCH_VERIFY_CERTS", "")
    search_mode = search_mode_override or _resolve_setting("ELASTICSEARCH_SEARCH_MODE", DEFAULT_SEARCH_MODE)

    verify_certs = None
    if verify_raw.lower() in ("false", "0", "no"):
        verify_certs = False
    elif verify_raw.lower() in ("true", "1", "yes"):
        verify_certs = True

    return ElasticsearchConfig.from_env(
        url=url,
        index_name=index_name,
        cloud_id=cloud_id,
        api_key=api_key,
        username=username,
        password=password,
        verify_certs=verify_certs,
        search_mode=search_mode,
    )


def setup_langsmith() -> tuple:
    """LangSmith(LangChain Tracing V2) 자동 추적 활성화 함수"""
    api_key = _resolve_setting("LANGCHAIN_API_KEY", DEFAULT_LANGCHAIN_API_KEY)
    if api_key:
        project = _resolve_setting("LANGCHAIN_PROJECT", DEFAULT_LANGCHAIN_PROJECT) or "cv-agent-elastic"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        return True, project
    return False, ""


# ==============================================================================
# 0-1. FastAPI 기반 BigQuery 로그 수집기 연동 함수 (비동기 전송)
# ==============================================================================
LOGGER_API_URL = os.environ.get("LOGGER_API_URL", "http://127.0.0.1:8000")


def send_click_log(button_name: str, extra_info: str = ""):
    """FastAPI 로그 수집기(/log/click)로 비동기 전송하여 BigQuery에 적재합니다."""
    user_id = st.session_state.get("user_id", "anonymous")
    session_id = st.session_state.get("thread_id", "default_session")
    try:
        import threading

        def _post():
            try:
                import requests

                requests.post(
                    f"{LOGGER_API_URL}/log/click",
                    json={
                        "session_id": session_id,
                        "user_id": user_id,
                        "button_name": button_name,
                        "page": "portfolio_assistant_elastic",
                        "extra_info": extra_info,
                    },
                    timeout=2.0,
                )
            except Exception:
                pass

        threading.Thread(target=_post, daemon=True).start()
    except Exception:
        pass


def send_chat_log(
    user_query: str,
    llm_response: str,
    node: str,
    model_name: str,
    docs_count: int,
    elapsed_seconds: float = 0.0,
    provider: str = "google_ai_studio",
):
    """FastAPI 로그 수집기(/log/chat)로 비동기 전송하여 BigQuery(chat_logs)에 적재합니다."""
    user_id = st.session_state.get("user_id", "anonymous")
    session_id = st.session_state.get("thread_id", "default_session")
    try:
        import threading

        def _post():
            try:
                import requests

                requests.post(
                    f"{LOGGER_API_URL}/log/chat",
                    json={
                        "session_id": session_id,
                        "user_id": user_id,
                        "user_query": user_query,
                        "llm_response": llm_response,
                        "node": node,
                        "model_name": model_name,
                        "provider": provider,
                        "retrieved_docs_count": docs_count,
                        "elapsed_seconds": elapsed_seconds,
                    },
                    timeout=3.0,
                )
            except Exception:
                pass

        threading.Thread(target=_post, daemon=True).start()
    except Exception:
        pass


def send_eval_log(user_query: str, eval_content: str, llm_response: str = "", target_type: str = "general"):
    """FastAPI 로그 수집기(/log/eval) 또는 직접 BigQuery 호출을 통해 eval_logs 테이블에 적재합니다."""
    user_id = st.session_state.get("user_id", "anonymous")
    session_id = st.session_state.get("thread_id", "default_session")
    try:
        import threading

        def _post():
            sent = False
            try:
                import requests

                res = requests.post(
                    f"{LOGGER_API_URL}/log/eval",
                    json={
                        "session_id": session_id,
                        "user_id": user_id,
                        "eval_content": eval_content,
                        "user_query": user_query,
                        "llm_response": llm_response,
                        "target_type": target_type,
                        "status": "NEW",
                    },
                    timeout=3.0,
                )
                if res.status_code == 200:
                    sent = True
            except Exception:
                pass

            if not sent:
                try:
                    from logger_api import insert_eval_log_direct

                    insert_eval_log_direct(
                        user_query=user_query,
                        eval_content=eval_content,
                        llm_response=llm_response,
                        session_id=session_id,
                        user_id=user_id,
                        target_type=target_type,
                        status="NEW",
                    )
                except Exception:
                    pass

        threading.Thread(target=_post, daemon=True).start()
    except Exception:
        pass


# ------------------------------------------------------------------------------
# 1. 페이지 환경설정 및 스타일
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="최규영 AI Portfolio Assistant (Elasticsearch RAG)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# 0-2. 상단바 프로젝트 링크 클릭 시 BigQuery 로그 수집 및 자동 리다이렉트 핸들러
# ------------------------------------------------------------------------------
PROJECT_NAV_LINKS = {
    "rag": {
        "name": "RAG 대시보드 프로젝트",
        "url": "https://ragdashboard-blue.vercel.app/",
        "button_id": "top_nav_link_rag_dashboard",
    },
    "restaurant": {
        "name": "스마트 맛집 추천 플랫폼",
        "url": "https://restaurant-nu-ruby.vercel.app/",
        "button_id": "top_nav_link_smart_restaurant",
    },
    "toystore": {
        "name": "모바일 매출 부진 분석",
        "url": "https://mobile-revenue-growth-strategy.vercel.app/",
        "button_id": "top_nav_link_toystore_revenue",
    },
}

out_param = st.query_params.get("out")
if out_param and out_param in PROJECT_NAV_LINKS:
    target = PROJECT_NAV_LINKS[out_param]
    session_id = st.session_state.get("thread_id", "top_nav_click_session")
    user_id = st.session_state.get("user_id", "anonymous")

    try:
        import requests

        requests.post(
            f"{LOGGER_API_URL}/log/click",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "button_name": target["button_id"],
                "page": "top_navbar",
                "extra_info": f"{target['name']} ({target['url']})",
            },
            timeout=2.0,
        )
    except Exception:
        try:
            from logger_api import CLICK_TABLE_ID, _insert_to_bigquery, get_current_time_info

            iso_ts, kst_str = get_current_time_info()
            _insert_to_bigquery(
                CLICK_TABLE_ID,
                {
                    "timestamp": iso_ts,
                    "kst_time": kst_str,
                    "session_id": session_id,
                    "user_id": user_id,
                    "button_name": target["button_id"],
                    "page": "top_navbar",
                    "extra_info": f"{target['name']} ({target['url']})",
                },
            )
        except Exception:
            pass

    st.markdown(
        f"""
        <meta http-equiv="refresh" content="0; url={target['url']}">
        <script>window.location.replace("{target['url']}");</script>
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 40px; text-align: center; color: #1E293B;">
            <h2>🚀 {target['name']} 웹사이트로 이동 중입니다...</h2>
            <p style="color: #64748B;">클릭 로그가 BigQuery에 기록되었습니다. 잠시 후 실제 배포 사이트로 자동 연결됩니다.</p>
            <p>자동으로 이동하지 않으면 <a href="{target['url']}" style="color: #2563EB; font-weight: bold;">여기를 클릭하세요</a>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .doc-pill {
        display: inline-block;
        padding: 4px 10px;
        background-color: #EEF2F6;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        color: #334155;
    }
    .badge-rag {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #BAE6FD;
    }
    .badge-chat {
        background-color: #F1F5F9;
        color: #475569;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #CBD5E1;
    }
    .badge-eval {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #FDE68A;
    }
    .api-status-box {
        padding: 10px 14px;
        border-radius: 8px;
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 2. 세션 상태 초기화 및 고정 API 키 할당
# ------------------------------------------------------------------------------
INITIAL_ASSISTANT_CONTENT = (
    "안녕하세요! 최규영 님의 포트폴리오 및 프로젝트 안내를 담당하는 **AI 어시스턴트 (Elasticsearch RAG)**입니다.\n\n"
    "본 시스템은 **Elasticsearch 검색 엔진(BM25 키워드 + Dense Vector kNN + Hybrid RRF)**을 기반으로 2가지 핵심 역할로 구분되어 동작합니다:\n\n"
    "1. 🔍 **포트폴리오 RAG봇 (Elasticsearch 문서 검색)**\n"
    "   - 최규영 님의 **기본 이력(학력·자격증·어학·기술 스택)** 및 3대 프로젝트(**RAG 대시보드, 스마트 맛집 추천 플랫폼, ToyStore 모바일 매출 분석**)에 대해 무엇이든 질문해 보세요! Elasticsearch에서 문서를 검색하여 정확한 근거를 바탕으로 답변합니다.\n\n"
    "2. 💬 **일반 채팅봇 (평가 & 피드백 수집용)**\n"
    "   - 프로젝트나 Agent의 답변 설명에 대해 아쉬운 점, 보완할 점, 솔직한 평가를 **`평가: `**로 시작하여 남겨주시면 일반 채팅봇이 이를 받아 **Google BigQuery**에 안전하게 적재합니다.\n"
    "   - 남겨주신 평가는 제가 추후 확인하여 Agent 답변 품질 개선 및 프로젝트 업데이트에 소중히 반영합니다.\n\n"
    "💡 **평가 남기기 예시**:\n"
    "> `평가: agent에게 프로젝트에 대해서 물어봤는데 설명이 아쉬워요`\n"
    "> `평가: Elasticsearch 하이브리드 검색 기반 답변이 정확해서 좋았습니다.`\n\n"
    "궁금한 점을 자유롭게 질문하시거나, 편하게 평가를 남겨주세요!"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": INITIAL_ASSISTANT_CONTENT,
            "node": "chat_bot",
            "docs": [],
        }
    ]

if "es_store" not in st.session_state:
    st.session_state.es_store = None

SYSTEM_API_KEY = resolve_api_key()
OPENROUTER_API_KEY = resolve_openrouter_api_key()
OPENROUTER_MODEL = resolve_openrouter_model()

st.session_state.current_api_key = SYSTEM_API_KEY
st.session_state.current_openrouter_key = OPENROUTER_API_KEY

has_any_key = bool(SYSTEM_API_KEY or OPENROUTER_API_KEY)
IS_LANGSMITH_ACTIVE, LANGSMITH_PROJECT_NAME = setup_langsmith()

if "user_id" not in st.session_state:
    st.session_state.user_id = f"usr_{uuid.uuid4().hex[:12]}"

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"sess_{uuid.uuid4().hex[:8]}"


# ------------------------------------------------------------------------------
# 3. 사이드바 구성
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 시스템 설정 및 상태")

    if SYSTEM_API_KEY and OPENROUTER_API_KEY:
        st.markdown(
            """
            <div class="api-status-box" style="background-color: #F0FDF4; border-color: #BBF7D0; color: #166534;">
                🟢 <b>Google AI Studio</b>: 정상 인증<br>
                🔄 <b>OpenRouter (Fallback)</b>: 정상 인증<br>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif SYSTEM_API_KEY:
        st.markdown(
            """
            <div class="api-status-box" style="background-color: #F0FDF4; border-color: #BBF7D0; color: #166534;">
                🟢 <b>Google AI Studio</b>: 정상 인증<br>
                <span style="font-size: 0.8rem; color: #64748B;">⚪ OpenRouter (Fallback): 미등록 (선택 사항)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif OPENROUTER_API_KEY:
        st.markdown(
            f"""
            <div class="api-status-box" style="background-color: #EFF6FF; border-color: #BFDBFE; color: #1D4ED8;">
                🔄 <b>OpenRouter</b>: 단독 인증 (<code>{OPENROUTER_MODEL}</code>)<br>
                <span style="font-size: 0.8rem; color: #64748B;">⚪ Google AI Studio: 미등록 (BM25 모드 지원)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("⚠️ 시스템에 등록된 API 키(Google AI Studio 또는 OpenRouter)가 없습니다.\n환경 변수 또는 st.secrets 설정을 확인해 주세요.")

    if IS_LANGSMITH_ACTIVE:
        st.markdown(
            f"""
            <div style="background-color: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #5B21B6; margin-bottom: 12px;">
                🦜️ <b>LangSmith Tracing</b>: 정상 연결<br>
                <span style="font-size: 0.73rem; color: #7C3AED;">• 프로젝트: <code>{LANGSMITH_PROJECT_NAME}</code></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 8px 12px; font-size: 0.78rem; color: #64748B; margin-bottom: 12px;">
                🦜️ <b>LangSmith Tracing</b>: 미연동<br>
                <span style="font-size: 0.72rem; color: #94A3B8;">(키 입력 시 LangGraph/LLM 자동 추적)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    model_name = "gemini-2.5-flash-lite"
    mode = "auto"
    top_k = 4

    fallback_badge = (
        f"<code>{OPENROUTER_MODEL}</code>"
        if OPENROUTER_API_KEY
        else "<span style='color: #94A3B8;'>미설정 (단일 엔진 모드)</span>"
    )

    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; font-size: 0.85em; color: #475569; margin-bottom: 12px;">
            <div style="font-weight: 600; color: #1E293B; margin-bottom: 6px;">🤖 AI 엔진 및 폴백 구성</div>
            <div>• <b>주(Primary) 모델</b>: <code>{model_name}</code> (Google AI)</div>
            <div style="margin-top: 3px;">• <b>보조(Fallback) 모델</b>: {fallback_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_mode = _resolve_setting("ELASTICSEARCH_SEARCH_MODE", DEFAULT_SEARCH_MODE).lower()
    mode_options = ["hybrid", "bm25", "vector"]
    default_idx = mode_options.index(default_mode) if default_mode in mode_options else 0

    search_mode = st.radio(
        "🔍 Elasticsearch 검색 방식",
        options=mode_options,
        format_func=lambda x: {
            "hybrid": "⚡ 하이브리드 (BM25 + kNN RRF)",
            "bm25": "🔤 BM25 키워드 풀텍스트",
            "vector": "🧭 kNN 벡터 유사도",
        }[x],
        index=default_idx,
        help="Hybrid는 키워드 검색(BM25)과 벡터 검색(kNN)을 RRF로 결합합니다. BM25는 임베딩 API 없이도 동작합니다.",
    )

    es_config = resolve_elasticsearch_config(search_mode_override=search_mode)

    st.divider()

    st.subheader("📦 Elasticsearch 인덱스 상태")
    if st.session_state.es_store is not None:
        try:
            status = st.session_state.es_store.get_index_status()
            vec_label = f"활성 ({status['vector_dims']}d)" if status["has_vectors"] else "BM25 전용"
            st.success(
                f"✅ 인덱스 활성화됨 (총 {status['doc_count']}개 청크)\n\n"
                f"- 인덱스: `{status['index_name']}`\n"
                f"- 클러스터: `{status['cluster_name']}`\n"
                f"- 분석기: `{status['analyzer']}` | 벡터: `{vec_label}`"
            )
        except Exception:
            st.warning("인덱스 상태 확인 중")
    else:
        st.info(f"🔄 Elasticsearch 연결 대기 중 (`{es_config.cloud_id or es_config.url}`)")

    with st.expander("📄 적재된 4개 문서 목록", expanded=False):
        for fname, meta in DOC_METADATA_CONFIG.items():
            st.markdown(f"**`{fname}`**\n- {meta['title']}\n- 분류: `{meta['category']}`")

    if st.button("🔄 Elasticsearch 인덱스 재구축", use_container_width=True):
        send_click_log("elasticsearch_rebuild_button", f"사이드바 인덱스 재구축 클릭 ({es_config.index_name})")
        with st.spinner("4개 문서를 다시 청킹하고 Elasticsearch 인덱스에 적재 중..."):
            try:
                store = get_or_create_elasticsearch_store(
                    es_config=es_config,
                    google_api_key=SYSTEM_API_KEY or None,
                    force_reload=True,
                    enable_vectors=bool(SYSTEM_API_KEY),
                )
                st.session_state.es_store = store
                st.success(f"재적재 완료! 총 {store.count()}개 청크")
                st.rerun()
            except Exception as e:
                st.error(f"재구축 실패: {e}")

    if st.button("🗑️ 대화 내역 초기화", use_container_width=True):
        send_click_log("reset_chat_history_button", "대화 내역 초기화 클릭")
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": INITIAL_ASSISTANT_CONTENT,
                "node": "chat_bot",
                "docs": [],
            }
        ]
        st.session_state.thread_id = f"sess_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 6px; padding: 8px 10px; font-size: 0.73rem; color: #64748B; margin-top: 10px;">
            👤 <b>방문자 ID</b>: <code>{st.session_state.user_id}</code><br>
            💬 <b>세션 ID</b>: <code>{st.session_state.thread_id}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# 4. Elasticsearch 스토어 및 LangGraph 초기화 로직 (Fallback 지원)
# ------------------------------------------------------------------------------
es_store = None
retriever = None
llm = None
langgraph_app = None
es_connection_error = None

if has_any_key:
    try:
        if st.session_state.es_store is None:
            with st.spinner("Elasticsearch 인덱스를 확인하고 문서를 로드하는 중..."):
                st.session_state.es_store = get_or_create_elasticsearch_store(
                    es_config=es_config,
                    google_api_key=SYSTEM_API_KEY or None,
                    force_reload=False,
                    enable_vectors=bool(SYSTEM_API_KEY),
                )

        es_store = st.session_state.es_store
        if es_store is not None:
            es_store.config.search_mode = search_mode
            retriever = es_store.as_retriever(search_kwargs={"k": top_k, "search_mode": search_mode})
    except Exception as e:
        es_connection_error = str(e)

    try:
        llm = init_llm_with_fallback(
            google_api_key=SYSTEM_API_KEY,
            openrouter_api_key=OPENROUTER_API_KEY,
            google_model=model_name,
            openrouter_model=OPENROUTER_MODEL,
            temperature=0.2,
        )
        langgraph_app = build_langgraph_app(llm, retriever)
    except Exception as e:
        st.error(f"LLM 초기화 중 오류가 발생했습니다: {e}")


# ------------------------------------------------------------------------------
# 5. 최상단 네비게이션 바 & 헤더
# ------------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("🌐 **프로젝트 라이브 웹사이트 바로가기** (클릭 시 로그 수집 후 새 탭에서 실제 배포 사이트로 이동)")
    top_col1, top_col2, top_col3 = st.columns(3)
    with top_col1:
        st.link_button(
            "📊 RAG 대시보드 바로가기 ↗",
            "?out=rag",
            use_container_width=True,
            help="자연어 기반 개인사업자 평가정보 RAG 대시보드 사이트로 이동합니다. (BigQuery 클릭 로그 수집)",
        )
    with top_col2:
        st.link_button(
            "🍽 스마트 맛집 추천 플랫폼 바로가기 ↗",
            "?out=restaurant",
            use_container_width=True,
            help="Canvas 혼잡도 수치화 & Redshift 스마트 맛집 추천 플랫폼으로 이동합니다. (BigQuery 클릭 로그 수집)",
        )
    with top_col3:
        st.link_button(
            "📈 모바일 매출 부진 분석 바로가기 ↗",
            "?out=toystore",
            use_container_width=True,
            help="ToyStore AARR 퍼널 분석 & 성장 개선 전략 사이트로 이동합니다. (BigQuery 클릭 로그 수집)",
        )

st.markdown('<div class="main-title">🔍 최규영 AI Portfolio Assistant (Elasticsearch RAG)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Google Gemini & Elasticsearch RAG (BM25 / Dense Vector kNN / Hybrid RRF), 그리고 LangGraph 상태 그래프(rag_bot & chat_bot)로 구동되는 포트폴리오 질의응답 챗봇입니다.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div>
      <span class="doc-pill">📄 PROFILE.md (기본정보·학력·자격증·역량)</span>
      <span class="doc-pill">📊 RAG_DASHBOARD.md (RAG 대시보드)</span>
      <span class="doc-pill">🍽 SMART_RESTAURANT.md (맛집 추천 플랫폼)</span>
      <span class="doc-pill">📈 TOYSTORE_MOBILE_REVENUE.md (ToyStore 분석)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if es_connection_error:
    st.error(
        f"⚠️ **Elasticsearch 연결 실패**: {es_connection_error}\n\n"
        "- 로컬 환경이라면 `cv_agent_elastic` 폴더에서 `docker compose up -d`로 Elasticsearch를 실행해 주세요.\n"
        "- Elastic Cloud 또는 원격 클러스터를 사용한다면 `.streamlit/secrets.toml` 또는 환경 변수(`ELASTICSEARCH_URL`, `ELASTICSEARCH_API_KEY` 등)를 확인해 주세요."
    )

if not has_any_key:
    st.warning(
        "⚠️ **AI API Key가 감지되지 않았습니다.**\n\n"
        "- **Google AI Studio API Key**: `.streamlit/secrets.toml`에 `GOOGLE_API_KEY = '...'` 또는 환경변수 설정\n"
        "- **OpenRouter API Key (Fallback)**: `.streamlit/secrets.toml`에 `OPENROUTER_API_KEY = '...'` 또는 환경변수 설정"
    )
elif not OPENROUTER_API_KEY:
    st.caption("💡 **안내**: 보조(Fallback) OpenRouter 키를 등록하시면 Google AI Studio 할당량 소진 시에도 답변이 중단되지 않고 자동 전환됩니다.")


# ------------------------------------------------------------------------------
# 6. 추천 질문 빠른 입력 버튼 (Quick Prompts)
# ------------------------------------------------------------------------------
prompt_to_send = None

st.caption("💡 추천 질문을 클릭하거나 평가 예시를 바로 테스트해 보세요:")
cols = st.columns(5)
with cols[0]:
    if st.button("🎓 학력·자격증·어학", use_container_width=True):
        send_click_log("quick_prompt_profile", extra_info="학력·자격증·어학")
        prompt_to_send = "최규영 님의 학력, 자격증, 어학 성적, 기술 스택을 각각 표로 깔끔하게 정리해줘."
with cols[1]:
    if st.button("📊 RAG 대시보드", use_container_width=True):
        send_click_log("quick_prompt_rag_dashboard", extra_info="RAG 대시보드")
        prompt_to_send = "RAG 대시보드 프로젝트의 문제점(AS-IS)과 하이브리드 필터링 해결책(TO-BE), 그리고 성과를 설명해줘."
with cols[2]:
    if st.button("🍽 스마트 맛집", use_container_width=True):
        send_click_log("quick_prompt_smart_restaurant", extra_info="스마트 맛집")
        prompt_to_send = "스마트 맛집 추천 플랫폼에서 Canvas 이미지를 OpenCV로 수치화한 원리와 트러블슈팅 사례를 알려줘."
with cols[3]:
    if st.button("📈 ToyStore 분석", use_container_width=True):
        send_click_log("quick_prompt_toystore", extra_info="ToyStore 분석")
        prompt_to_send = "ToyStore 모바일 매출 분석에서 AARR 프레임워크 진단 결과와 2대 핵심 개선 레버는 무엇인가요?"
with cols[4]:
    if st.button("📝 평가 남기기 예시", use_container_width=True):
        send_click_log("quick_prompt_eval", extra_info="평가 남기기 예시")
        prompt_to_send = "평가: agent에게 프로젝트에 대해서 물어봤는데 설명이 아쉬워요"

st.divider()


# ------------------------------------------------------------------------------
# 7. 대화 히스토리 출력
# ------------------------------------------------------------------------------
def _render_retrieved_docs(docs):
    with st.expander(f"🔍 Elasticsearch 검색 청크 ({len(docs)}개) 및 점수·메타데이터 보기", expanded=False):
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "출처 미상")
            sec = doc.metadata.get("section", "섹션")
            chunk_id = doc.metadata.get("chunk_id", "")
            method = doc.metadata.get("retrieval_method", "elasticsearch")
            score = doc.metadata.get("es_score")
            score_str = f" | **점수**: `{score}`" if score is not None else ""
            st.markdown(
                f"**[{i}] 출처**: `{src}` | **섹션**: `{sec}` | `{chunk_id}` | **방식**: `{method}`{score_str}"
            )
            st.text(doc.page_content.strip())
            st.markdown("---")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "node" in msg:
            if "평가" in msg["node"]:
                st.markdown('<span class="badge-eval">📝 chat_bot (평가/피드백 접수 & BigQuery 적재)</span>', unsafe_allow_html=True)
            elif "quota" in msg["node"]:
                st.markdown('<span class="badge-eval" style="background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D;">⚠️ API 토큰/할당량 전체 소모 안내</span>', unsafe_allow_html=True)
            elif "OpenRouter" in msg["node"]:
                if "rag" in msg["node"]:
                    st.markdown('<span class="badge-rag" style="background-color: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE;">🔍 rag_bot (Elasticsearch 검색 · OpenRouter 폴백)</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-chat" style="background-color: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE;">💬 chat_bot (일반 대화 · OpenRouter 폴백)</span>', unsafe_allow_html=True)
            elif "rag" in msg["node"]:
                st.markdown('<span class="badge-rag">🔍 rag_bot (Elasticsearch 문서 검색 기반)</span>', unsafe_allow_html=True)
            elif "chat" in msg["node"]:
                st.markdown('<span class="badge-chat">💬 chat_bot (일반 대화)</span>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        if msg.get("docs"):
            _render_retrieved_docs(msg["docs"])


# ------------------------------------------------------------------------------
# 8. 사용자 입력 처리
# ------------------------------------------------------------------------------
user_input = st.chat_input("최규영 님의 포트폴리오를 질문하거나, '평가: ...' 형태로 피드백을 남겨보세요...")

if prompt_to_send:
    user_input = prompt_to_send

if user_input:
    if not has_any_key or not langgraph_app:
        st.error("API Key(Google AI Studio 또는 OpenRouter)가 연결되지 않아 질문을 처리할 수 없습니다. 시스템 설정을 확인해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Elasticsearch에서 관련 문서를 검색하고 답변을 생성하는 중..."):
                start_t = time.time()
                try:
                    answer_text, executed_node, retrieved_docs = run_query(
                        graph=langgraph_app,
                        llm=llm,
                        retriever=retriever,
                        user_query=user_input,
                        thread_id=st.session_state.thread_id,
                        mode=mode,
                    )
                    elapsed = time.time() - start_t

                    is_fallback_used = (
                        "OpenRouter" in executed_node
                        or getattr(llm, "last_fallback_occurred", False)
                        or getattr(llm, "last_provider_used", "") == "fallback"
                    )

                    if is_fallback_used:
                        st.toast(
                            f"⚡ Google AI Studio 할당량이 소진되어 OpenRouter({OPENROUTER_MODEL}) 보조 모델로 안전하게 응답했습니다.",
                            icon="🔄",
                        )

                    eval_info = extract_evaluation_info(user_input)
                    if eval_info:
                        send_eval_log(
                            user_query=user_input,
                            eval_content=eval_info["eval_content"],
                            llm_response=answer_text,
                            target_type=eval_info["target_type"],
                        )
                        st.toast("📝 소중한 평가가 BigQuery(eval_logs)에 성공적으로 적재되었습니다!", icon="✅")

                    recorded_model = (
                        f"{OPENROUTER_MODEL} (OpenRouter Fallback)"
                        if is_fallback_used
                        else f"{model_name} (Google AI Studio)"
                    )
                    current_provider = "openrouter" if is_fallback_used else "google_ai_studio"
                    send_chat_log(
                        user_query=user_input,
                        llm_response=answer_text,
                        node=f"{executed_node} [es:{search_mode}]",
                        model_name=recorded_model,
                        provider=current_provider,
                        docs_count=len(retrieved_docs) if retrieved_docs else 0,
                        elapsed_seconds=round(elapsed, 2),
                    )

                    if "평가" in executed_node:
                        st.markdown('<span class="badge-eval">📝 chat_bot (평가/피드백 접수 & BigQuery 적재)</span>', unsafe_allow_html=True)
                    elif "quota" in executed_node:
                        st.markdown('<span class="badge-eval" style="background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D;">⚠️ API 토큰/할당량 전체 소모 안내</span>', unsafe_allow_html=True)
                        st.warning("⚠️ Google AI Studio 및 OpenRouter API의 토큰/할당량이 모두 소모되었습니다. 잠시 후 다시 시도해 주세요.")
                    elif "OpenRouter" in executed_node:
                        if "rag" in executed_node:
                            st.markdown('<span class="badge-rag" style="background-color: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE;">🔍 rag_bot (Elasticsearch 검색 · OpenRouter 폴백)</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-chat" style="background-color: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE;">💬 chat_bot (일반 대화 · OpenRouter 폴백)</span>', unsafe_allow_html=True)
                    elif "rag" in executed_node:
                        st.markdown('<span class="badge-rag">🔍 rag_bot (Elasticsearch 문서 검색 기반)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-chat">💬 chat_bot (일반 대화)</span>', unsafe_allow_html=True)

                    st.markdown(answer_text)

                    if retrieved_docs:
                        _render_retrieved_docs(retrieved_docs)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "node": executed_node,
                            "docs": retrieved_docs,
                        }
                    )

                except Exception as e:
                    elapsed = time.time() - start_t
                    if is_quota_exceeded_error(e):
                        answer_text = QUOTA_EXHAUSTED_MESSAGE
                        executed_node = "quota_exceeded"
                        st.markdown('<span class="badge-eval" style="background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D;">⚠️ API 토큰/할당량 전체 소모 안내</span>', unsafe_allow_html=True)
                        st.warning("⚠️ Google AI Studio 및 OpenRouter API의 토큰/할당량이 모두 소모되었습니다. 잠시 후 다시 시도해 주세요.")
                        st.markdown(answer_text)

                        send_chat_log(
                            user_query=user_input,
                            llm_response=answer_text,
                            node=executed_node,
                            model_name=f"{model_name} / {OPENROUTER_MODEL} (전체 소진)",
                            provider="all_quota_exceeded",
                            docs_count=0,
                            elapsed_seconds=round(elapsed, 2),
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer_text,
                                "node": executed_node,
                                "docs": [],
                            }
                        )
                    else:
                        error_msg = f"답변 생성 중 일시적인 오류가 발생했습니다: {e}\n\n잠시 후 다시 시도해 주시기 바랍니다."
                        st.error(error_msg)
                        send_chat_log(
                            user_query=user_input,
                            llm_response=error_msg,
                            node="error",
                            model_name=model_name,
                            provider="error",
                            docs_count=0,
                            elapsed_seconds=round(elapsed, 2),
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_msg,
                                "node": "error",
                                "docs": [],
                            }
                        )
