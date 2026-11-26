"""
app2.py (cv_agent_elastic - 배포/고정 API 키 + Elasticsearch + OpenRouter 폴백 + BigQuery 로거 버전)

실제 Elasticsearch 8.x(HNSW kNN + 한국어 Nori/BM25) 기반 배포용 Streamlit 애플리케이션입니다.
- 사용자에게 API 키를 요구하지 않고 환경변수 / st.secrets의 Google AI Studio & OpenRouter & Elasticsearch 설정 사용
- FastAPI 로그 수집기(logger_api.py)와 연동하여 click_logs / chat_logs / eval_logs를 BigQuery에 비동기 적재
- Docker Compose / Cloud Run 배포 최적화
"""

import os
import time
import streamlit as st
from rag_engine import (
    DEFAULT_ES_INDEX,
    DEFAULT_ES_URL,
    DOC_METADATA_CONFIG,
    QUOTA_EXHAUSTED_MESSAGE,
    ElasticsearchConfig,
    build_langgraph_app,
    extract_evaluation_info,
    get_or_create_es_store,
    init_llm_with_fallback,
    is_quota_exceeded_error,
    run_query,
)

# ==============================================================================
# 0. 고정 설정 (환경변수 또는 st.secrets가 없을 경우 폴백)
# ==============================================================================
DEFAULT_GOOGLE_API_KEY = ""
DEFAULT_OPENROUTER_API_KEY = ""
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_LANGCHAIN_API_KEY = ""
DEFAULT_LANGCHAIN_PROJECT = "cv-agent-elastic"


def _resolve_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    env_val = os.environ.get(key, "").strip()
    if env_val:
        return env_val
    return default.strip()


def resolve_es_config() -> ElasticsearchConfig:
    return ElasticsearchConfig(
        url=_resolve_secret("ELASTICSEARCH_URL", DEFAULT_ES_URL),
        index_name=_resolve_secret("ELASTICSEARCH_INDEX", DEFAULT_ES_INDEX),
        cloud_id=_resolve_secret("ELASTICSEARCH_CLOUD_ID", ""),
        api_key=_resolve_secret("ELASTICSEARCH_API_KEY", ""),
        username=_resolve_secret("ELASTICSEARCH_USERNAME", ""),
        password=_resolve_secret("ELASTICSEARCH_PASSWORD", ""),
        verify_certs=_resolve_secret("ELASTICSEARCH_VERIFY_CERTS", "false").lower() == "true",
    )


def setup_langsmith() -> tuple:
    api_key = _resolve_secret("LANGCHAIN_API_KEY", DEFAULT_LANGCHAIN_API_KEY)
    if api_key:
        project = _resolve_secret("LANGCHAIN_PROJECT", DEFAULT_LANGCHAIN_PROJECT) or "cv-agent-elastic"
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
# 1. 페이지 환경설정 및 리다이렉트 핸들러
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="최규영 AI Portfolio Assistant (Elasticsearch Hybrid RAG)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
                "page": "top_navbar_elastic",
                "extra_info": f"{target['name']} ({target['url']})",
            },
            timeout=2.0,
        )
    except Exception:
        try:
            from logger_api import _insert_to_bigquery, CLICK_TABLE_ID, get_current_time_info
            iso_ts, kst_str = get_current_time_info()
            _insert_to_bigquery(CLICK_TABLE_ID, {
                "timestamp": iso_ts,
                "kst_time": kst_str,
                "session_id": session_id,
                "user_id": user_id,
                "button_name": target["button_id"],
                "page": "top_navbar_elastic",
                "extra_info": f"{target['name']} ({target['url']})",
            })
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
        background-color: #DCFCE7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #86EFAC;
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
# 2. 세션 상태 초기화 및 환경변수 해석
# ------------------------------------------------------------------------------
INITIAL_ASSISTANT_CONTENT = (
    "안녕하세요! **실제 Elasticsearch 하이브리드 검색 엔진**으로 구동되는 최규영 님의 포트폴리오 **AI 어시스턴트**입니다.\n\n"
    "본 시스템은 2가지 핵심 역할로 구분되어 동작합니다:\n\n"
    "1. 🔍 **Elasticsearch RAG봇 (정보 확인용)**\n"
    "   - 최규영 님의 **기본 이력(학력·자격증·어학·기술 스택)** 및 3대 프로젝트(**RAG 대시보드, 스마트 맛집 추천 플랫폼, ToyStore 모바일 매출 분석**)를 **Elasticsearch HNSW kNN + 한국어 BM25(Nori) + RRF + Cross-Encoder**로 검색해 답변합니다.\n\n"
    "2. 💬 **일반 채팅봇 (평가 & 피드백 수집용)**\n"
    "   - **`평가: `**로 시작하여 피드백을 남겨주시면 **Google BigQuery**에 실시간 적재됩니다.\n\n"
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

SYSTEM_API_KEY = _resolve_secret("GOOGLE_API_KEY", DEFAULT_GOOGLE_API_KEY)
OPENROUTER_API_KEY = _resolve_secret("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_API_KEY)
OPENROUTER_MODEL = _resolve_secret("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
ES_CONFIG = resolve_es_config()
has_any_key = bool(SYSTEM_API_KEY or OPENROUTER_API_KEY)

IS_LANGSMITH_ACTIVE, LANGSMITH_PROJECT_NAME = setup_langsmith()

if "user_id" not in st.session_state:
    import uuid
    st.session_state.user_id = f"usr_{uuid.uuid4().hex[:12]}"

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = f"sess_{uuid.uuid4().hex[:8]}"


# ------------------------------------------------------------------------------
# 3. 사이드바 구성
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 시스템 설정 및 상태")

    if SYSTEM_API_KEY and OPENROUTER_API_KEY:
        st.markdown(
            """
            <div class="api-status-box">
                🟢 <b>Google AI Studio</b>: 정상 인증<br>
                🔄 <b>OpenRouter (Fallback)</b>: 정상 인증
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif SYSTEM_API_KEY:
        st.markdown(
            """
            <div class="api-status-box">
                🟢 <b>Google AI Studio</b>: 정상 인증<br>
                <span style="font-size: 0.8rem; color: #64748B;">⚪ OpenRouter (Fallback): 미등록</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif OPENROUTER_API_KEY:
        st.markdown(
            f"""
            <div class="api-status-box" style="background-color: #EFF6FF; border-color: #BFDBFE; color: #1D4ED8;">
                🔄 <b>OpenRouter</b>: 단독 인증 (<code>{OPENROUTER_MODEL}</code>)<br>
                <span style="font-size: 0.8rem; color: #64748B;">⚪ Google AI Studio: 미등록</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("⚠️ API 키(GOOGLE_API_KEY 또는 OPENROUTER_API_KEY)가 설정되지 않았습니다.")

    if IS_LANGSMITH_ACTIVE:
        st.markdown(
            f"""
            <div style="background-color: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #5B21B6; margin-bottom: 12px;">
                🦜️ <b>LangSmith Tracing</b>: 정상 연결 (<code>{LANGSMITH_PROJECT_NAME}</code>)
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
        else "<span style='color: #94A3B8;'>미설정</span>"
    )

    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; font-size: 0.85em; color: #475569; margin-bottom: 12px;">
            <div style="font-weight: 600; color: #1E293B; margin-bottom: 6px;">🔍 Elasticsearch 4단계 RAG 구성</div>
            <div>• <b>주(Primary) 모델</b>: <code>{model_name}</code></div>
            <div style="margin-top: 3px;">• <b>보조(Fallback)</b>: {fallback_badge}</div>
            <div style="margin-top: 3px;">• <b>ES 엔드포인트</b>: <code>{ES_CONFIG.cloud_id or ES_CONFIG.url}</code></div>
            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px dashed #CBD5E1; font-size: 0.92em; color: #334155;">
                ① 쿼리 플래너 (Query Planner)<br>
                ② Elasticsearch HNSW kNN + BM25<br>
                ③ RRF 알고리즘 (Rank Fusion)<br>
                ④ 크로스 인코더 리랭킹 (Reranker)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.subheader("📦 Elasticsearch 인덱스 상태")
    if st.session_state.es_store is not None:
        try:
            total_chunks = st.session_state.es_store.count()
            es_ver = st.session_state.es_store.get_cluster_version()
            analyzer = st.session_state.es_store.analyzer_mode
            dims = st.session_state.es_store.embedding_dims
            st.success(
                f"✅ ES v{es_ver} (`{st.session_state.es_store.index_name}`)\n\n"
                f"- 청크: **{total_chunks}개** ({dims}d) | 분석기: `{analyzer}`"
            )
        except Exception:
            st.warning("Elasticsearch 인덱스 상태 확인 중")
    else:
        st.info("🔄 Elasticsearch 인덱스 연결 대기 중")

    with st.expander("📄 적재된 4개 문서 목록", expanded=False):
        for fname, meta in DOC_METADATA_CONFIG.items():
            st.markdown(f"**`{fname}`**\n- {meta['title']}\n- 분류: `{meta['category']}`")

    if st.button("🔄 Elasticsearch 인덱스 재구축", use_container_width=True):
        send_click_log("es_index_rebuild_button", "사이드바 Elasticsearch 인덱스 재구축 클릭")
        if not SYSTEM_API_KEY:
            st.error("Elasticsearch 재구축에는 GOOGLE_API_KEY(임베딩용)가 필요합니다.")
        else:
            with st.spinner("4개 문서를 다시 청킹·임베딩하여 Elasticsearch에 적재 중..."):
                try:
                    store = get_or_create_es_store(
                        api_key=SYSTEM_API_KEY,
                        es_config=ES_CONFIG,
                        force_reload=True,
                    )
                    st.session_state.es_store = store
                    st.success(f"재적재 완료! 총 {store.count()}개 청크 ({store.analyzer_mode})")
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
        import uuid
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
# 4. Elasticsearch 스토어 및 LangGraph 초기화
# ------------------------------------------------------------------------------
es_retriever = None
llm = None
langgraph_app = None

if has_any_key:
    try:
        if st.session_state.es_store is None and SYSTEM_API_KEY:
            with st.spinner("Elasticsearch 클러스터에 연결하고 인덱스를 확인하는 중..."):
                st.session_state.es_store = get_or_create_es_store(
                    api_key=SYSTEM_API_KEY,
                    es_config=ES_CONFIG,
                    force_reload=False,
                )

        if st.session_state.es_store is not None:
            es_retriever = st.session_state.es_store.as_retriever(search_kwargs={"k": top_k})

        cache_sig = f"{SYSTEM_API_KEY}:{OPENROUTER_API_KEY}:{model_name}:{OPENROUTER_MODEL}:{top_k}:{ES_CONFIG.cache_signature()}"
        if st.session_state.get("_graph_cache_sig") != cache_sig or st.session_state.get("langgraph_app") is None:
            st.session_state.llm = init_llm_with_fallback(
                google_api_key=SYSTEM_API_KEY,
                openrouter_api_key=OPENROUTER_API_KEY,
                google_model=model_name,
                openrouter_model=OPENROUTER_MODEL,
                temperature=0.2,
            )
            st.session_state.langgraph_app = build_langgraph_app(st.session_state.llm, es_retriever)
            st.session_state._graph_cache_sig = cache_sig

        llm = st.session_state.llm
        langgraph_app = st.session_state.langgraph_app
    except Exception as e:
        st.error(f"초기화 중 오류가 발생했습니다: {e}")


# ------------------------------------------------------------------------------
# 5. 최상단 네비게이션 바 & 헤더
# ------------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("🌐 **프로젝트 라이브 웹사이트 바로가기** (클릭 시 로그 수집 후 새 탭에서 실제 배포 사이트로 이동)")
    top_col1, top_col2, top_col3 = st.columns(3)
    with top_col1:
        st.link_button("📊 RAG 대시보드 바로가기 ↗", "?out=rag", use_container_width=True)
    with top_col2:
        st.link_button("🍽 스마트 맛집 추천 플랫폼 바로가기 ↗", "?out=restaurant", use_container_width=True)
    with top_col3:
        st.link_button("📈 모바일 매출 부진 분석 바로가기 ↗", "?out=toystore", use_container_width=True)

st.markdown('<div class="main-title">🔍 최규영 AI Portfolio Assistant (Elasticsearch Hybrid RAG)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    '<b>1단계: 쿼리 플래너</b> ➔ <b>2단계: Elasticsearch HNSW kNN + BM25(Nori)</b> ➔ <b>3단계: RRF 알고리즘</b> ➔ <b>4단계: 크로스 인코더 리랭킹</b> 파이프라인과 LangGraph로 구동되는 포트폴리오 AI 어시스턴트입니다.'
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


# ------------------------------------------------------------------------------
# 6. 추천 질문 버튼
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


def _render_retrieved_docs_expander(docs):
    if not docs:
        return
    with st.expander(f"🔍 Elasticsearch 4단계 RAG 파이프라인 결과 & 참고 청크 ({len(docs)}개) 보기", expanded=False):
        plan_summary = docs[0].metadata.get("query_plan_summary")
        if plan_summary:
            lat = plan_summary.get("latencies_ms") or {}
            lat_str = ""
            if lat:
                lat_str = (
                    f"\n- ⚡ **단계별 소요 시간**: 1단계 플래너 `{lat.get('stage1_planner_ms', 0)}ms` | "
                    f"2단계 ES kNN+BM25 `{lat.get('stage2_hybrid_ms', 0)}ms` | "
                    f"3단계 RRF `{lat.get('stage3_rrf_ms', 0)}ms` | "
                    f"4단계 Rerank `{lat.get('stage4_rerank_ms', 0)}ms` "
                    f"(총 검색 `{lat.get('total_retrieval_ms', 0)}ms`)"
                )
            st.markdown(
                f"**[Elasticsearch 인덱스: `{plan_summary.get('es_index', DEFAULT_ES_INDEX)}` | 분석기: `{plan_summary.get('es_analyzer', 'unknown')}`]**\n"
                f"- **1단계 플래너 (`{plan_summary.get('planner_mode', 'fast')}`)**: `{plan_summary.get('standalone_query', '')}`\n"
                f"- **BM25 키워드**: `{', '.join(plan_summary.get('bm25_keywords') or [])}`\n"
                f"- **질의 의도 / 타겟 문서**: `{plan_summary.get('query_intent', '')}` / `{plan_summary.get('target_sources', [])}`"
                f"{lat_str}"
            )
            st.markdown("---")
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "출처 미상")
            sec = doc.metadata.get("section", "섹션")
            chunk_id = doc.metadata.get("chunk_id", "")
            vec_rank = doc.metadata.get("vector_rank", "-")
            bm25_rank = doc.metadata.get("bm25_rank", "-")
            rrf_score = doc.metadata.get("rrf_score", "-")
            rerank_score = doc.metadata.get("rerank_score", "-")
            st.markdown(
                f"**[{i}] 출처**: `{src}` | **섹션**: `{sec}` | `{chunk_id}`  \n"
                f"📊 **2단계(ES kNN 순위 / ES BM25 순위)**: `{vec_rank}` / `{bm25_rank}` ➔ "
                f"**3단계(RRF 점수)**: `{rrf_score}` ➔ **4단계(Cross-Encoder 점수)**: `{rerank_score}`"
            )
            st.text(doc.page_content.strip())
            st.markdown("---")


# ------------------------------------------------------------------------------
# 7. 대화 히스토리 출력
# ------------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "node" in msg:
            if "평가" in msg["node"]:
                st.markdown('<span class="badge-eval">📝 chat_bot (평가/피드백 접수 & BigQuery 적재)</span>', unsafe_allow_html=True)
            elif "quota" in msg["node"]:
                st.markdown('<span class="badge-eval">⚠️ API 토큰/할당량 전체 소모 안내</span>', unsafe_allow_html=True)
            elif "OpenRouter" in msg["node"]:
                if "rag" in msg["node"]:
                    st.markdown('<span class="badge-rag" style="background-color: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE;">🔍 rag_bot (Elasticsearch Hybrid · OpenRouter 폴백)</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-chat" style="background-color: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE;">💬 chat_bot (일반 대화 · OpenRouter 폴백)</span>', unsafe_allow_html=True)
            elif "rag" in msg["node"]:
                st.markdown('<span class="badge-rag">🔍 rag_bot (Elasticsearch kNN+BM25 ➔ RRF ➔ Cross-Encoder)</span>', unsafe_allow_html=True)
            elif "chat" in msg["node"]:
                st.markdown('<span class="badge-chat">💬 chat_bot (일반 대화)</span>', unsafe_allow_html=True)

        st.markdown(msg["content"])
        if msg.get("docs"):
            _render_retrieved_docs_expander(msg["docs"])


# ------------------------------------------------------------------------------
# 8. 사용자 입력 처리
# ------------------------------------------------------------------------------
user_input = st.chat_input("최규영 님의 포트폴리오를 질문하거나, '평가: ...' 형태로 피드백을 남겨보세요...")

if prompt_to_send:
    user_input = prompt_to_send

if user_input:
    if not has_any_key or not langgraph_app:
        st.error("API Key 또는 Elasticsearch 연결이 준비되지 않았습니다. 환경 변수 설정을 확인해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Elasticsearch 4단계 RAG 파이프라인(쿼리 플래너 ➔ ES kNN+BM25 ➔ RRF ➔ 리랭킹) 실행 중..."):
                start_t = time.time()
                try:
                    answer_text, executed_node, retrieved_docs = run_query(
                        graph=langgraph_app,
                        llm=llm,
                        es_store=es_retriever,
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
                        st.toast(f"⚡ OpenRouter({OPENROUTER_MODEL}) 보조 모델로 안전하게 응답했습니다.", icon="🔄")

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
                        node=executed_node,
                        model_name=recorded_model,
                        provider=current_provider,
                        docs_count=len(retrieved_docs) if retrieved_docs else 0,
                        elapsed_seconds=round(elapsed, 2),
                    )

                    if "평가" in executed_node:
                        st.markdown('<span class="badge-eval">📝 chat_bot (평가/피드백 접수 & BigQuery 적재)</span>', unsafe_allow_html=True)
                    elif "quota" in executed_node:
                        st.markdown('<span class="badge-eval">⚠️ API 토큰/할당량 전체 소모 안내</span>', unsafe_allow_html=True)
                    elif "OpenRouter" in executed_node:
                        if "rag" in executed_node:
                            st.markdown('<span class="badge-rag" style="background-color: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE;">🔍 rag_bot (Elasticsearch Hybrid · OpenRouter 폴백)</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-chat" style="background-color: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE;">💬 chat_bot (일반 대화 · OpenRouter 폴백)</span>', unsafe_allow_html=True)
                    elif "rag" in executed_node:
                        st.markdown('<span class="badge-rag">🔍 rag_bot (Elasticsearch kNN+BM25 ➔ RRF ➔ Cross-Encoder)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-chat">💬 chat_bot (일반 대화)</span>', unsafe_allow_html=True)

                    st.markdown(answer_text)
                    if retrieved_docs:
                        _render_retrieved_docs_expander(retrieved_docs)

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
                        st.warning("⚠️ API 토큰/할당량이 모두 소모되었습니다.")
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
                    else:
                        answer_text = f"답변 생성 중 일시적인 오류가 발생했습니다: {e}"
                        executed_node = "error"
                        st.error(answer_text)
                        send_chat_log(
                            user_query=user_input,
                            llm_response=answer_text,
                            node="error",
                            model_name=model_name,
                            provider="error",
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
