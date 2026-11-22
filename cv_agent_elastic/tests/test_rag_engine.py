"""
tests/test_rag_engine.py

cv_agent_elastic/rag_engine.py의 문서 청킹, Elasticsearch 인덱스 매핑/적재,
BM25 / Vector(kNN) / Hybrid(RRF) 검색, 메타데이터 필터 보강 검색,
임베딩 장애 시 BM25 폴백, LangGraph 라우팅을 검증하는 단위 테스트입니다.
(외부 Elasticsearch 서버나 실제 API 키 없이 인메모리 Mock ES 클라이언트로 검증합니다.)
"""

import math
import os
import sys
import unittest
from typing import Any, Dict, List, Optional

# cv_agent_elastic 루트를 import 경로에 추가
ELASTIC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ELASTIC_DIR not in sys.path:
    sys.path.insert(0, ELASTIC_DIR)

# 테스트 환경에 elasticsearch 패키지가 미설치된 경우를 대비한 경량 스텁 지원
try:
    import elasticsearch  # noqa: F401
    HAS_ES_PACKAGE = True
except ImportError:
    import types

    HAS_ES_PACKAGE = False
    es_mod = types.ModuleType("elasticsearch")

    class _DummyElasticsearch:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyHelpers:
        @staticmethod
        def bulk(client, actions):
            count = 0
            for act in actions:
                client.index(index=act["_index"], id=act["_id"], document=act["_source"])
                count += 1
            return count, []

    es_mod.Elasticsearch = _DummyElasticsearch
    es_mod.helpers = _DummyHelpers
    sys.modules["elasticsearch"] = es_mod

import rag_engine  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402


class FakeEmbeddings:
    """결정론적 4차원 임베딩 생성기 (테스트용)"""

    def __init__(self, fail_on_query: bool = False):
        self.fail_on_query = fail_on_query

    @staticmethod
    def _embed(text: str) -> List[float]:
        t = text.lower()
        groups = [
            ["rag", "대시보드", "bigquery", "개인사업자", "chromadb", "하이브리드"],
            ["맛집", "opencv", "canvas", "카카오맵", "redshift", "airflow", "혼잡도"],
            ["toystore", "모바일", "aarr", "퍼널", "cvr", "매출", "desktop"],
            ["학력", "자격증", "sqld", "adsp", "토익", "프로필", "연락처"],
        ]
        v = [0.1 + sum(t.count(k) for k in grp) for grp in groups]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self.fail_on_query:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: embedding quota exceeded")
        return self._embed(text)


class FakeESIndices:
    def __init__(self, parent: "FakeESClient"):
        self.parent = parent

    def exists(self, index: str) -> bool:
        return index in self.parent.indices_data

    def create(self, index: str, settings: Dict[str, Any], mappings: Dict[str, Any]):
        self.parent.indices_data[index] = {}
        self.parent.indices_settings[index] = settings
        self.parent.indices_mappings[index] = mappings
        return {"acknowledged": True}

    def delete(self, index: str):
        self.parent.indices_data.pop(index, None)
        self.parent.indices_settings.pop(index, None)
        self.parent.indices_mappings.pop(index, None)
        return {"acknowledged": True}

    def refresh(self, index: str):
        return {"_shards": {"successful": 1}}

    def get_mapping(self, index: str) -> Dict[str, Any]:
        return {index: {"mappings": self.parent.indices_mappings.get(index, {})}}


class FakeESCat:
    def __init__(self, plugins: Optional[List[Dict[str, str]]] = None):
        self._plugins = plugins or []

    def plugins(self, format: str = "json"):
        return self._plugins


class FakeESClient:
    """Elasticsearch 8.x API를 인메모리로 모사하는 테스트용 클라이언트"""

    def __init__(self, plugins: Optional[List[Dict[str, str]]] = None):
        self.indices_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.indices_settings: Dict[str, Dict[str, Any]] = {}
        self.indices_mappings: Dict[str, Dict[str, Any]] = {}
        self.indices = FakeESIndices(self)
        self.cat = FakeESCat(plugins=plugins)

    def ping(self) -> bool:
        return True

    def info(self) -> Dict[str, Any]:
        return {"cluster_name": "mock-es-cluster", "version": {"number": "8.15.3"}}

    def index(self, index: str, id: str, document: Dict[str, Any]):
        self.indices_data.setdefault(index, {})[id] = document
        return {"result": "created"}

    def count(self, index: str) -> Dict[str, Any]:
        return {"count": len(self.indices_data.get(index, {}))}

    @staticmethod
    def _matches_filters(doc: Dict[str, Any], filters: List[Dict[str, Any]]) -> bool:
        for f in filters:
            term = f.get("term", {})
            for field, val in term.items():
                if doc.get(field) != val:
                    return False
        return True

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def search(
        self,
        index: str,
        query: Optional[Dict[str, Any]] = None,
        knn: Optional[Dict[str, Any]] = None,
        size: int = 10,
        sort: Optional[List[Dict[str, Any]]] = None,
        _source: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        docs = self.indices_data.get(index, {})
        excludes = set((_source or {}).get("excludes", []))

        def _strip(doc: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in doc.items() if k not in excludes}

        hits = []

        if knn is not None:
            qvec = knn["query_vector"]
            filters = knn.get("filter", [])
            for doc_id, doc in docs.items():
                if filters and not self._matches_filters(doc, filters):
                    continue
                if "embedding" not in doc:
                    continue
                score = (1.0 + self._cosine(qvec, doc["embedding"])) / 2.0
                hits.append({"_id": doc_id, "_score": score, "_source": _strip(doc)})
            hits.sort(key=lambda h: h["_score"], reverse=True)
            return {"hits": {"hits": hits[:size]}}

        query = query or {"match_all": {}}
        if "match_all" in query:
            for doc_id, doc in docs.items():
                hits.append({"_id": doc_id, "_score": 1.0, "_source": _strip(doc)})
        elif "bool" in query:
            bool_q = query["bool"]
            filters = bool_q.get("filter", [])
            must = bool_q.get("must", [])
            qtext = ""
            if must and "multi_match" in must[0]:
                qtext = must[0]["multi_match"]["query"].lower()
            tokens = [tok for tok in qtext.replace(",", " ").replace("?", " ").split() if tok]

            for doc_id, doc in docs.items():
                if filters and not self._matches_filters(doc, filters):
                    continue
                if not tokens:
                    hits.append({"_id": doc_id, "_score": 1.0, "_source": _strip(doc)})
                    continue

                haystack = " ".join(
                    [
                        str(doc.get("page_content", "")),
                        str(doc.get("section", "")),
                        str(doc.get("doc_title", "")),
                    ]
                ).lower()
                score = 0.0
                for tok in tokens:
                    if tok in haystack:
                        score += haystack.count(tok)
                if score > 0:
                    hits.append({"_id": doc_id, "_score": float(score), "_source": _strip(doc)})

        if sort:
            sort_field = next(iter(sort[0].keys()))
            hits.sort(key=lambda h: str(h["_source"].get(sort_field, "")))
        else:
            hits.sort(key=lambda h: h["_score"], reverse=True)

        return {"hits": {"hits": hits[:size]}}


class FakeLLM:
    """LangGraph 라우팅 및 응답 생성을 검증하는 Mock LLM"""

    def __init__(self):
        self.prompts: List[Any] = []

    def invoke(self, prompt: Any, **kwargs: Any) -> AIMessage:
        self.prompts.append(prompt)
        text = str(prompt)
        if "반드시 docs 또는 chat 이라는 단어 하나만 출력하세요" in text:
            if any(k in text for k in ["OpenCV", "학력", "프로젝트", "ToyStore", "RAG"]):
                return AIMessage(content="docs")
            return AIMessage(content="chat")
        if "[포트폴리오 문서 자료]" in text:
            return AIMessage(content="Elasticsearch 검색 문서 기반 답변입니다.")
        return AIMessage(content="일반 대화 / 평가 접수 응답입니다.")


class TestElasticsearchRAGEngine(unittest.TestCase):
    def setUp(self):
        self.orig_bulk = rag_engine.helpers.bulk
        rag_engine.helpers.bulk = self._mock_bulk
        self.chunks = rag_engine.load_and_split_documents(root_dir=ELASTIC_DIR)
        self.client = FakeESClient()
        self.embeddings = FakeEmbeddings()
        self.store = rag_engine.ElasticsearchRAGStore(
            client=self.client,
            index_name="test_portfolio_index",
            embeddings=self.embeddings,
            config=rag_engine.ElasticsearchConfig(index_name="test_portfolio_index", search_mode="hybrid"),
        )
        self.store.index_documents(self.chunks, force_reload=True, enable_vectors=True)

    def tearDown(self):
        rag_engine.helpers.bulk = self.orig_bulk

    @staticmethod
    def _mock_bulk(client, actions):
        count = 0
        for act in actions:
            client.index(index=act["_index"], id=act["_id"], document=act["_source"])
            count += 1
        return count, []

    def test_load_and_split_documents_covers_all_four_files(self):
        sources = {c.metadata["source"] for c in self.chunks}
        self.assertEqual(
            sources,
            {"PROFILE.md", "RAG_DASHBOARD.md", "SMART_RESTAURANT.md", "TOYSTORE_MOBILE_REVENUE.md"},
        )
        self.assertGreater(len(self.chunks), 10)

    def test_index_creation_and_dense_vector_mapping(self):
        self.assertEqual(self.store.count(), len(self.chunks))
        self.assertTrue(self.store.has_vectors)
        self.assertEqual(self.store.vector_dims, 4)
        self.assertEqual(self.store.analyzer_name, "cjk")

        status = self.store.get_index_status()
        self.assertEqual(status["cluster_name"], "mock-es-cluster")
        self.assertEqual(status["doc_count"], len(self.chunks))

    def test_nori_analyzer_auto_detection(self):
        nori_client = FakeESClient(plugins=[{"component": "analysis-nori"}])
        self.assertEqual(rag_engine.detect_korean_analyzer(nori_client), "nori")
        settings, mappings = rag_engine.build_index_settings_and_mappings("nori", vector_dims=8)
        self.assertIn("analysis", settings)
        self.assertEqual(mappings["properties"]["page_content"]["analyzer"], "portfolio_text_analyzer")
        self.assertEqual(mappings["properties"]["embedding"]["dims"], 8)

    def test_bm25_search_finds_relevant_project(self):
        docs = self.store.search_bm25("OpenCV Canvas 혼잡도", k=4)
        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0].metadata["source"], "SMART_RESTAURANT.md")
        self.assertEqual(docs[0].metadata["retrieval_method"], "bm25")

    def test_vector_knn_search_finds_relevant_project(self):
        docs = self.store.search_vector("ToyStore 모바일 매출 AARR 퍼널", k=4)
        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0].metadata["source"], "TOYSTORE_MOBILE_REVENUE.md")
        self.assertEqual(docs[0].metadata["retrieval_method"], "vector (kNN)")

    def test_hybrid_rrf_search_combines_bm25_and_vector(self):
        docs = self.store.search_hybrid("OpenCV Canvas 혼잡도 카카오맵", k=4)
        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0].metadata["source"], "SMART_RESTAURANT.md")
        self.assertEqual(docs[0].metadata["retrieval_method"], "hybrid (BM25 + kNN RRF)")
        self.assertGreater(docs[0].metadata["es_score"], 0)

    def test_hybrid_falls_back_to_bm25_when_embedding_quota_exhausted(self):
        self.store.embeddings = FakeEmbeddings(fail_on_query=True)
        docs = self.store.search_hybrid("OpenCV Canvas 혼잡도", k=4)
        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0].metadata["source"], "SMART_RESTAURANT.md")
        self.assertEqual(docs[0].metadata["retrieval_method"], "bm25 (vector fallback)")

    def test_smart_retrieval_project_intro_and_profile(self):
        retriever = self.store.as_retriever(search_kwargs={"k": 4, "search_mode": "hybrid"})

        proj_docs = rag_engine.retrieve_relevant_documents(retriever, "진행한 프로젝트들 간단히 소개해줘")
        proj_sources = {d.metadata["source"] for d in proj_docs}
        self.assertEqual(
            proj_sources,
            {"RAG_DASHBOARD.md", "SMART_RESTAURANT.md", "TOYSTORE_MOBILE_REVENUE.md"},
        )

        prof_docs = rag_engine.retrieve_relevant_documents(retriever, "학력과 자격증, 어학 성적 알려줘")
        self.assertTrue(any(d.metadata["source"] == "PROFILE.md" for d in prof_docs))

    def test_langgraph_end_to_end_routing_and_rag(self):
        retriever = self.store.as_retriever(search_kwargs={"k": 4, "search_mode": "hybrid"})
        llm = FakeLLM()
        graph = rag_engine.build_langgraph_app(llm, retriever)

        # 1. 문서 질문 -> rag_bot
        ans, node, docs = rag_engine.run_query(
            graph=graph,
            llm=llm,
            retriever=retriever,
            user_query="스마트 맛집에서 OpenCV로 Canvas 이미지를 어떻게 수치화했나요?",
            thread_id="test_thread_1",
            mode="auto",
        )
        self.assertEqual(node, "rag_bot")
        self.assertIn("Elasticsearch 검색 문서 기반 답변", ans)
        self.assertGreater(len(docs), 0)

        # 2. 평가 접두사 -> 즉시 chat_bot (Fast-Path)
        ans_eval, node_eval, docs_eval = rag_engine.run_query(
            graph=graph,
            llm=llm,
            retriever=retriever,
            user_query="평가: Elasticsearch 버전 설명이 아주 명확해요",
            thread_id="test_thread_1",
            mode="auto",
        )
        self.assertIn("평가/피드백", node_eval)
        self.assertEqual(docs_eval, [])


if __name__ == "__main__":
    unittest.main()
