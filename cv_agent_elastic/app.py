"""
app.py

Streamlit 기반의 ChatGPT 스타일 웹 인터페이스 애플리케이션입니다 (Elasticsearch RAG 버전).
- Google AI Studio API 키 및 Elasticsearch 연결 정보를 사용자가 직접 설정
- 4개 문서(PROFILE.md, RAG_DASHBOARD.md, SMART_RESTAURANT.md, TOYSTORE_MOBILE_REVENUE.md)를
  Elasticsearch 인덱스에 적재하고 BM25 / Vector(kNN) / Hybrid(RRF) 검색 수행
- LangGraph를 활용한 문서 기반 Q&A (rag_bot) 및 일반 대화 (chat_bot) 라우팅
"""

import os
import uuid
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_engine import (
    DEFAULT_ES_INDEX,
    DEFAULT_ES_URL,
    DOC_METADATA_CONFIG,
    ElasticsearchConfig,
    build_langgraph_app,
    extract_evaluation_info,
    get_or_create_elasticsearch_store,
    run_query,
)

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

    try:
        from datetime import datetime, timezone
        from logger_api import CLICK_TABLE_ID, _insert_to_bigquery

        _insert_to_bigquery(
            CLICK_TABLE_ID,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 2. 세션 상태 초기화
# ------------------------------------------------------------------------------
INITIAL_ASSISTANT_CONTENT = (
    "안녕하세요! 최규영 님의 포트폴리오 및 프로젝트 안내를 담당하는 **AI 어시스턴트 (Elasticsearch RAG 버전)**입니다.\n\n"
    "본 시스템은 **Elasticsearch 검색 엔진(BM25 키워드 + Dense Vector kNN + Hybrid RRF)**을 기반으로 2가지 핵심 역할로 구분되어 동작합니다:\n\n"
    "1. 🔍 **포트폴리오 RAG봇 (Elasticsearch 문서 검색)**\n"
    "   - 최규영 님의 **기본 이력(학력·자격증·어학·기술 스택)** 및 3대 프로젝트(**RAG 대시보드, 스마트 맛집 추천 플랫폼, ToyStore 모바일 매출 분석**)에 대해 무엇이든 질문해 보세요! Elasticsearch로 문서를 검색하여 정확한 근거를 바탕으로 답변합니다.\n\n"
    "2. 💬 **일반 채팅봇 (평가 & 피드백 수집용)**\n"
    "   - 프로젝트나 Agent의 답변 설명에 대해 아쉬운 점, 보완할 점, 솔직한 평가를 **`평가: `**로 시작하여 남겨주시면 일반 채팅봇이 이를 받아 **Google BigQuery**에 안전하게 적재합니다.\n\n"
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

if "es_signature" not in st.session_state:
    st.session_state.es_signature = None

if "current_api_key" not in st.session_state:
    st.session_state.current_api_key = ""

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session_01"


# ------------------------------------------------------------------------------
# 3. 사이드바 구성 (API 키, Elasticsearch 연결, 검색 모드, 인덱스 관리)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 설정 및 상태")

    st.subheader("🔑 Google AI Studio API Key")
    env_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "API 키를 입력하세요",
        value=st.session_state.current_api_key or env_key,
        type="password",
        help="Google AI Studio(aistudio.google.com)에서 발급받은 Gemini API 키를 입력하세요.",
    )

    if not api_key_input:
        st.info("💡 [Google AI Studio에서 API 키 발급받기 ↗](https://aistudio.google.com/app/apikey)")
    else:
        st.session_state.current_api_key = api_key_input.strip()

    st.divider()

    st.subheader("🔍 Elasticsearch 연결 설정")
    es_url = st.text_input(
        "Elasticsearch URL",
        value=os.environ.get("ELASTICSEARCH_URL", DEFAULT_ES_URL),
        help="예: http://localhost:9200 또는 Elastic Cloud 엔드포인트",
    )
    es_index = st.text_input(
        "인덱스 이름",
        value=os.environ.get("ELASTICSEARCH_INDEX", DEFAULT_ES_INDEX),
    )

    with st.expander("🔐 인증 설정 (선택 사항)", expanded=False):
        es_cloud_id = st.text_input(
            "Elastic Cloud ID",
            value=os.environ.get("ELASTICSEARCH_CLOUD_ID", ""),
            help="Elastic Cloud 사용 시 Cloud ID 입력 (입력 시 URL보다 우선)",
        )
        es_api_key = st.text_input(
            "Elasticsearch API Key",
            value=os.environ.get("ELASTICSEARCH_API_KEY", ""),
            type="password",
        )
        es_username = st.text_input(
            "Username (Basic Auth)",
            value=os.environ.get("ELASTICSEARCH_USERNAME", ""),
        )
        es_password = st.text_input(
            "Password (Basic Auth)",
            value=os.environ.get("ELASTICSEARCH_PASSWORD", ""),
            type="password",
        )

    st.divider()

    st.subheader("🤖 모델 및 검색 모드 선택")
    model_name = st.selectbox(
        "Gemini 모델",
        options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="답변 생성에 사용할 Gemini 모델을 선택합니다.",
    )

    search_mode = st.radio(
        "Elasticsearch 검색 방식",
        options=["hybrid", "bm25", "vector"],
        format_func=lambda x: {
            "hybrid": "⚡ 하이브리드 (BM25 + kNN RRF)",
            "bm25": "🔤 BM25 키워드 풀텍스트",
            "vector": "🧭 kNN 벡터 유사도",
        }[x],
        index=0,
        help="Hybrid는 키워드 매칭(BM25)과 의미 벡터 검색(kNN)을 RRF로 결합합니다.",
    )

    mode = st.radio(
        "대화 모드",
        options=["auto", "rag_only", "chat_only"],
        format_func=lambda x: {
            "auto": "🔀 자동 라우팅 (LangGraph Auto)",
            "rag_only": "📚 포트폴리오 RAG 모드 (rag_bot)",
            "chat_only": "💬 일반 대화 모드 (chat_bot)",
        }[x],
        index=0,
        help="자동 라우팅은 질문의 성격에 따라 rag_bot과 chat_bot으로 분기합니다.",
    )

    top_k = st.slider("검색 청크 수 (Top-k)", min_value=2, max_value=8, value=4)

    st.divider()

    st.subheader("📦 Elasticsearch 인덱스 상태")
    if st.session_state.es_store is not None:
        try:
            status = st.session_state.es_store.get_index_status()
            vec_label = f"활성 ({status['vector_dims']}d)" if status["has_vectors"] else "BM25 전용"
            st.success(
                f"✅ 인덱스 활성화됨 (총 {status['doc_count']}개 청크)\n\n"
                f"- 클러스터: `{status['cluster_name']}`\n"
                f"- 분석기: `{status['analyzer']}` | 벡터: `{vec_label}`"
            )
        except Exception:
            st.warning("인덱스 상태 확인 중")
    else:
        st.warning("⚠️ API 키 및 Elasticsearch 연결 후 인덱스가 초기화됩니다.")

    with st.expander("📄 적재된 4개 문서 목록", expanded=False):
        for fname, meta in DOC_METADATA_CONFIG.items():
            st.markdown(f"**`{fname}`**\n- {meta['title']}\n- 분류: `{meta['category']}`")

    es_config = ElasticsearchConfig.from_env(
        url=es_url,
        index_name=es_index,
        cloud_id=es_cloud_id,
        api_key=es_api_key,
        username=es_username,
        password=es_password,
        search_mode=search_mode,
    )

    if st.button("🔄 Elasticsearch 인덱스 재구축", use_container_width=True):
        with st.spinner("4개 문서를 다시 청킹하고 Elasticsearch 인덱스에 적재 중..."):
            try:
                store = get_or_create_elasticsearch_store(
                    es_config=es_config,
                    google_api_key=st.session_state.current_api_key or None,
                    force_reload=True,
                    enable_vectors=(search_mode != "bm25"),
                )
                st.session_state.es_store = store
                st.session_state.es_signature = (
                    es_config.url,
                    es_config.cloud_id,
                    es_config.index_name,
                    bool(st.session_state.current_api_key),
                )
                st.success(f"재적재 완료! 총 {store.count()}개 청크")
                st.rerun()
            except Exception as e:
                st.error(f"재구축 실패: {e}")

    if st.button("🗑️ 대화 내역 초기화", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": INITIAL_ASSISTANT_CONTENT,
                "node": "chat_bot",
                "docs": [],
            }
        ]
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ------------------------------------------------------------------------------
# 4. Elasticsearch 스토어 및 LangGraph 초기화 로직
# ------------------------------------------------------------------------------
es_store = None
retriever = None
llm = None
langgraph_app = None

current_signature = (
    es_config.url,
    es_config.cloud_id,
    es_config.index_name,
    bool(st.session_state.current_api_key),
)

if st.session_state.current_api_key:
    try:
        if st.session_state.es_store is None or st.session_state.es_signature != current_signature:
            with st.spinner("Elasticsearch 인덱스를 확인하고 포트폴리오 문서를 로드하는 중..."):
                st.session_state.es_store = get_or_create_elasticsearch_store(
                    es_config=es_config,
                    google_api_key=st.session_state.current_api_key,
                    force_reload=False,
                    enable_vectors=(search_mode != "bm25"),
                )
                st.session_state.es_signature = current_signature

        es_store = st.session_state.es_store
        es_store.config.search_mode = search_mode
        retriever = es_store.as_retriever(search_kwargs={"k": top_k, "search_mode": search_mode})

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=st.session_state.current_api_key,
            temperature=0.2,
        )

        langgraph_app = build_langgraph_app(llm, retriever)
    except Exception as e:
        st.error(f"Elasticsearch 또는 LLM 초기화 중 오류가 발생했습니다: {e}")


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

if not st.session_state.current_api_key:
    with st.container(border=True):
        st.markdown("### 🔑 Google AI Studio API Key 등록")
        st.write("챗봇을 시작하려면 Gemini API 키가 필요합니다. 아래 입력창에 키를 입력하고 **[등록 및 시작]** 버튼을 눌러주세요.")

        c1, c2 = st.columns([4, 1])
        with c1:
            main_key_input = st.text_input(
                "API 키 입력",
                type="password",
                placeholder="AIzaSy... 로 시작하는 Gemini API Key를 붙여넣으세요",
                label_visibility="collapsed",
                key="main_key_field",
            )
        with c2:
            submit_btn = st.button("등록 및 시작", use_container_width=True, type="primary")

        if (submit_btn and main_key_input) or (main_key_input and main_key_input.startswith("AIza")):
            st.session_state.current_api_key = main_key_input.strip()
            st.success("API 키가 등록되었습니다! 화면을 갱신합니다...")
            st.rerun()

        st.caption("💡 API 키가 없으신가요? [Google AI Studio에서 무료로 키 발급받기 ↗](https://aistudio.google.com/app/apikey)")
else:
    st.caption(f"✅ Google AI Studio API Key 연결됨 | Elasticsearch 검색 모드: `{search_mode.upper()}`")


# ------------------------------------------------------------------------------
# 6. 추천 질문 빠른 입력 버튼 (Quick Prompts)
# ------------------------------------------------------------------------------
prompt_to_send = None

st.caption("💡 추천 질문을 클릭하거나 평가 예시를 바로 테스트해 보세요:")
cols = st.columns(5)
with cols[0]:
    if st.button("🎓 학력·자격증·어학", use_container_width=True):
        prompt_to_send = "최규영 님의 학력, 자격증, 어학 성적, 기술 스택을 표로 깔끔하게 정리해줘."
with cols[1]:
    if st.button("📊 RAG 대시보드", use_container_width=True):
        prompt_to_send = "RAG 대시보드 프로젝트의 문제점(AS-IS)과 하이브리드 필터링 해결책(TO-BE), 그리고 성과를 설명해줘."
with cols[2]:
    if st.button("🍽 스마트 맛집", use_container_width=True):
        prompt_to_send = "스마트 맛집 추천 플랫폼에서 Canvas 이미지를 OpenCV로 수치화한 원리와 트러블슈팅 사례를 알려줘."
with cols[3]:
    if st.button("📈 ToyStore 분석", use_container_width=True):
        prompt_to_send = "ToyStore 모바일 매출 분석에서 AARR 프레임워크 진단 결과와 2대 핵심 개선 레버는 무엇인가요?"
with cols[4]:
    if st.button("📝 평가 남기기 예시", use_container_width=True):
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
    if not st.session_state.current_api_key or langgraph_app is None:
        st.error("Google AI Studio API Key 또는 Elasticsearch 연결이 완료되지 않았습니다. 사이드바 설정을 확인해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Elasticsearch에서 관련 문서를 검색하고 답변을 생성하는 중..."):
                try:
                    answer_text, executed_node, retrieved_docs = run_query(
                        graph=langgraph_app,
                        llm=llm,
                        retriever=retriever,
                        user_query=user_input,
                        thread_id=st.session_state.thread_id,
                        mode=mode,
                    )

                    eval_info = extract_evaluation_info(user_input)
                    if eval_info:
                        try:
                            from logger_api import insert_eval_log_direct

                            insert_eval_log_direct(
                                user_query=user_input,
                                eval_content=eval_info["eval_content"],
                                llm_response=answer_text,
                                session_id=st.session_state.thread_id,
                                target_type=eval_info["target_type"],
                                status="NEW",
                            )
                        except Exception:
                            pass
                        st.toast("📝 소중한 평가가 BigQuery(eval_logs)에 성공적으로 적재되었습니다!", icon="✅")

                    if "평가" in executed_node:
                        st.markdown('<span class="badge-eval">📝 chat_bot (평가/피드백 접수 & BigQuery 적재)</span>', unsafe_allow_html=True)
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
                    error_msg = f"답변 생성 중 오류가 발생했습니다: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "node": "error",
                            "docs": [],
                        }
                    )
