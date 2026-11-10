"""
app.py

Streamlit 기반의 ChatGPT 스타일 웹 인터페이스 애플리케이션입니다.
- Google AI Studio API 키를 사용자가 직접 입력
- root 폴더 4개 문서(PROFILE.md, RAG_DASHBOARD.md, SMART_RESTAURANT.md, TOYSTORE_MOBILE_REVENUE.md)를 적재한 ChromaDB 기반 RAG
- LangGraph를 활용한 문서 기반 Q&A (rag_bot) 및 일반 대화 (chat_bot) 라우팅
"""

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_engine import (
    DOC_METADATA_CONFIG,
    ROOT_DIR,
    build_langgraph_app,
    extract_evaluation_info,
    get_or_create_vectorstore,
    load_and_split_documents,
    run_query,
)

# ------------------------------------------------------------------------------
# 1. 페이지 환경설정 및 스타일
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="최규영 AI Portfolio Assistant (RAG & Chat)",
    page_icon="🤖",
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
    
    # BigQuery로 클릭 이벤트 적재
    try:
        from logger_api import _insert_to_bigquery, CLICK_TABLE_ID
        from datetime import datetime, timezone
        _insert_to_bigquery(CLICK_TABLE_ID, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "button_name": target["button_id"],
            "page": "top_navbar",
            "extra_info": f"{target['name']} ({target['url']})",
        })
    except Exception:
        pass

    # 해당 프로젝트 공식 웹사이트로 즉시 리다이렉트
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
    .top-nav-bar {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 6px;
        margin-bottom: 16px;
    }
    .top-nav-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .nav-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 10px;
    }
    .nav-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        text-decoration: none !important;
        color: #1E293B !important;
        transition: all 0.2s ease-in-out;
    }
    .nav-card:hover {
        border-color: #3B82F6;
        background-color: #EFF6FF;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        color: #1D4ED8 !important;
    }
    .nav-card-content {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .nav-card-icon {
        font-size: 1.3rem;
    }
    .nav-card-text {
        display: flex;
        flex-direction: column;
    }
    .nav-card-name {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0F172A;
    }
    .nav-card-sub {
        font-size: 0.73rem;
        color: #64748B;
    }
    .nav-card-arrow {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 700;
    }
    .nav-card:hover .nav-card-arrow {
        color: #2563EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 2. 세션 상태 초기화
# ------------------------------------------------------------------------------
INITIAL_ASSISTANT_CONTENT = (
    "안녕하세요! 최규영 님의 포트폴리오 및 프로젝트 안내를 담당하는 **AI 어시스턴트**입니다.\n\n"
    "본 시스템은 2가지 핵심 역할로 구분되어 동작합니다:\n\n"
    "1. 📚 **포트폴리오 RAG봇 (정보 확인용)**\n"
    "   - 최규영 님의 **기본 이력(학력·자격증·어학·기술 스택)** 및 3대 프로젝트(**RAG 대시보드, 스마트 맛집 추천 플랫폼, ToyStore 모바일 매출 분석**)에 대해 무엇이든 질문해 보세요! 문서를 검색하여 정확한 근거를 바탕으로 답변합니다.\n\n"
    "2. 💬 **일반 채팅봇 (평가 & 피드백 수집용)**\n"
    "   - 프로젝트나 Agent의 답변 설명에 대해 아쉬운 점, 보완할 점, 솔직한 평가를 **`평가: `**로 시작하여 남겨주시면 일반 채팅봇이 이를 받아 **Google BigQuery**에 안전하게 적재합니다.\n"
    "   - 남겨주신 평가는 제가 추후 확인하여 Agent 답변 품질 개선 및 프로젝트 업데이트에 소중히 반영합니다.\n\n"
    "💡 **평가 남기기 예시**:\n"
    "> `평가: agent에게 프로젝트에 대해서 물어봤는데 설명이 아쉬워요`\n"
    "> `평가: RAG 대시보드의 하이브리드 필터링 설계 설명이 명확해서 좋았습니다.`\n\n"
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

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_api_key" not in st.session_state:
    st.session_state.current_api_key = ""

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session_01"


# ------------------------------------------------------------------------------
# 3. 사이드바 구성 (API 키, 모델 선택, 모드 설정, ChromaDB 관리)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 설정 및 상태")

    st.subheader("🔑 Google AI Studio API Key")
    # 환경변수에 기본값이 있다면 로드
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

    st.subheader("🤖 모델 및 모드 선택")
    model_name = st.selectbox(
        "Gemini 모델",
        options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="답변 생성에 사용할 Gemini 모델을 선택합니다.",
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

    st.subheader("📦 ChromaDB 컬렉션 상태")
    if st.session_state.vectorstore is not None:
        try:
            total_chunks = st.session_state.vectorstore._collection.count()
            st.success(f"✅ 컬렉션 활성화됨 (총 {total_chunks}개 청크)")
        except Exception:
            st.warning("컬렉션 상태 확인 중")
    else:
        st.warning("⚠️ API 키 입력 후 벡터스토어가 초기화됩니다.")

    # 4개 문서 요약 정보
    with st.expander("📄 적재된 4개 문서 목록", expanded=False):
        for fname, meta in DOC_METADATA_CONFIG.items():
            st.markdown(f"**`{fname}`**\n- {meta['title']}\n- 분류: `{meta['category']}`")

    # ChromaDB 재구축 버튼
    if st.button("🔄 ChromaDB 컬렉션 재구축", use_container_width=True):
        if not st.session_state.current_api_key:
            st.error("API 키를 먼저 입력해 주세요.")
        else:
            with st.spinner("4개 문서를 다시 청킹하고 ChromaDB에 적재 중..."):
                try:
                    vs = get_or_create_vectorstore(
                        api_key=st.session_state.current_api_key,
                        force_reload=True,
                    )
                    st.session_state.vectorstore = vs
                    st.success(f"재적재 완료! 총 {vs._collection.count()}개 청크")
                    st.rerun()
                except Exception as e:
                    st.error(f"재구축 실패: {e}")

    # 대화 초기화 버튼
    if st.button("🗑️ 대화 내역 초기화", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": INITIAL_ASSISTANT_CONTENT,
                "node": "chat_bot",
                "docs": [],
            }
        ]
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ------------------------------------------------------------------------------
# 4. 벡터스토어 및 LangGraph 초기화 로직
# ------------------------------------------------------------------------------
vectorstore = None
retriever = None
llm = None
langgraph_app = None

if st.session_state.current_api_key:
    try:
        if st.session_state.vectorstore is None:
            with st.spinner("ChromaDB 컬렉션을 확인하고 문서를 로드하는 중..."):
                st.session_state.vectorstore = get_or_create_vectorstore(
                    api_key=st.session_state.current_api_key,
                    force_reload=False,
                )

        vectorstore = st.session_state.vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=st.session_state.current_api_key,
            temperature=0.2,
        )

        langgraph_app = build_langgraph_app(llm, retriever)
    except Exception as e:
        st.error(f"초기화 중 오류가 발생했습니다: {e}")


# ------------------------------------------------------------------------------
# 5. 최상단 네비게이션 바 (프로젝트 3개 웹사이트 연결 바로가기) & 헤더
# ------------------------------------------------------------------------------
# 🌐 화면 맨 위 최상단 바: 프로젝트 3개 웹사이트 바로가기 버튼 3개 나란히 배치
with st.container(border=True):
    st.markdown("🌐 **프로젝트 라이브 웹사이트 바로가기** (클릭 시 로그 수집 후 새 탭에서 실제 배포 사이트로 이동)")
    top_col1, top_col2, top_col3 = st.columns(3)
    with top_col1:
        st.link_button(
            "📊 RAG 대시보드 바로가기 ↗",
            "?out=rag",
            use_container_width=True,
            help="자연어 기반 개인사업자 평가정보 RAG 대시보드 사이트로 이동합니다. (BigQuery 클릭 로그 수집)"
        )
    with top_col2:
        st.link_button(
            "🍽 스마트 맛집 추천 플랫폼 바로가기 ↗",
            "?out=restaurant",
            use_container_width=True,
            help="Canvas 혼잡도 수치화 & Redshift 스마트 맛집 추천 플랫폼으로 이동합니다. (BigQuery 클릭 로그 수집)"
        )
    with top_col3:
        st.link_button(
            "📈 모바일 매출 부진 분석 바로가기 ↗",
            "?out=toystore",
            use_container_width=True,
            help="ToyStore AARR 퍼널 분석 & 성장 개선 전략 사이트로 이동합니다. (BigQuery 클릭 로그 수집)"
        )

st.markdown('<div class="main-title">🤖 최규영 AI Portfolio Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Google Gemini & ChromaDB RAG, 그리고 LangGraph 상태 그래프(rag_bot & chat_bot)로 구동되는 포트폴리오 질의응답 챗봇입니다.'
    '</div>',
    unsafe_allow_html=True,
)

# 4개 문서 뱃지 표시
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

# API 키 미입력 시 메인 화면에 명확한 입력 카드 제공
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
                key="main_key_field"
            )
        with c2:
            submit_btn = st.button("등록 및 시작", use_container_width=True, type="primary")

        if (submit_btn and main_key_input) or (main_key_input and main_key_input.startswith("AIza")):
            st.session_state.current_api_key = main_key_input.strip()
            st.success("API 키가 등록되었습니다! 화면을 갱신합니다...")
            st.rerun()

        st.caption("💡 API 키가 없으신가요? [Google AI Studio에서 무료로 키 발급받기 ↗](https://aistudio.google.com/app/apikey)")
else:
    st.caption("✅ Google AI Studio API Key가 연결되어 있습니다. (키 변경은 왼쪽 사이드바에서 가능합니다)")


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
# 7. 대화 히스토리 출력 (ChatGPT 스타일)
# ------------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 실행 노드 뱃지 표시
        if msg["role"] == "assistant" and "node" in msg:
            if "평가" in msg["node"]:
                st.markdown('<span class="badge-eval">📝 chat_bot (평가/피드백 접수 & BigQuery 적재)</span>', unsafe_allow_html=True)
            elif "rag" in msg["node"]:
                st.markdown('<span class="badge-rag">📚 rag_bot (문서 검색 기반)</span>', unsafe_allow_html=True)
            elif "chat" in msg["node"]:
                st.markdown('<span class="badge-chat">💬 chat_bot (일반 대화)</span>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        # 참고한 문서 출처가 있으면 expander로 표시
        if msg.get("docs"):
            with st.expander(f"📚 참고한 문서 청크 ({len(msg['docs'])}개) 및 메타데이터 보기", expanded=False):
                for i, doc in enumerate(msg["docs"], 1):
                    src = doc.metadata.get("source", "출처 미상")
                    sec = doc.metadata.get("section", "섹션")
                    chunk_id = doc.metadata.get("chunk_id", "")
                    st.markdown(f"**[{i}] 출처**: `{src}` | **섹션**: `{sec}` | `{chunk_id}`")
                    st.text(doc.page_content.strip())
                    st.markdown("---")


# ------------------------------------------------------------------------------
# 8. 사용자 입력 처리 (Chat Input 또는 추천 질문 클릭)
# ------------------------------------------------------------------------------
user_input = st.chat_input("최규영 님의 포트폴리오를 질문하거나, '평가: ...' 형태로 피드백을 남겨보세요...")

if prompt_to_send:
    user_input = prompt_to_send

if user_input:
    if not st.session_state.current_api_key:
        st.error("Google AI Studio API Key가 설정되지 않았습니다. 사이드바에 키를 입력해 주세요.")
    else:
        # 사용자 메시지 세션에 추가 및 UI 렌더링
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 어시스턴트 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("생각하는 중..."):
                try:
                    answer_text, executed_node, retrieved_docs = run_query(
                        graph=langgraph_app,
                        llm=llm,
                        retriever=retriever,
                        user_query=user_input,
                        thread_id=st.session_state.thread_id,
                        mode=mode,
                    )

                    # 평가 쿼리 여부 확인 및 BigQuery 적재
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
                        st.toast("📝 소중한 평가가 BigQuery(eval_logs)에 성공적으로 적재되었습니다! (추후 업데이트에 반영됩니다)", icon="✅")

                    # 노드 뱃지 출력
                    if "평가" in executed_node:
                        st.markdown('<span class="badge-eval">📝 chat_bot (평가/피드백 접수 & BigQuery 적재)</span>', unsafe_allow_html=True)
                    elif "rag" in executed_node:
                        st.markdown('<span class="badge-rag">📚 rag_bot (문서 검색 기반)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-chat">💬 chat_bot (일반 대화)</span>', unsafe_allow_html=True)

                    st.markdown(answer_text)

                    # 검색 청크 표시
                    if retrieved_docs:
                        with st.expander(f"📚 참고한 문서 청크 ({len(retrieved_docs)}개) 및 메타데이터 보기", expanded=False):
                            for i, doc in enumerate(retrieved_docs, 1):
                                src = doc.metadata.get("source", "출처 미상")
                                sec = doc.metadata.get("section", "섹션")
                                chunk_id = doc.metadata.get("chunk_id", "")
                                st.markdown(f"**[{i}] 출처**: `{src}` | **섹션**: `{sec}` | `{chunk_id}`")
                                st.text(doc.page_content.strip())
                                st.markdown("---")

                    # 세션에 어시스턴트 메시지 저장
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
