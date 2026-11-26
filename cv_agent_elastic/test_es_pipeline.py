"""
test_es_pipeline.py

실제 Elasticsearch 서버가 꺼져 있는 환경에서도
cv_agent_elastic의 인덱스 매핑, Nori 탐지, ES kNN DSL, ES BM25 DSL,
4단계 하이브리드 파이프라인, LangGraph 상태 그래프 전체 흐름을 검증하는 단위 테스트입니다.
"""

import os
import unittest
from unittest.mock import MagicMock

os.environ["DISABLE_NEURAL_CROSS_ENCODER"] = "true"

from langchain_core.messages import AIMessage
import rag_engine as re_es


class _FakeEmbeddings:
    def __init__(self, dims: int = 8):
        self.dims = dims
        self.query_calls = 0

    def embed_query(self, text: str):
        self.query_calls += 1
        return [0.1] * self.dims

    def embed_documents(self, texts):
        return [[0.1] * self.dims for _ in texts]


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt, **kwargs):
        self.calls.append(prompt)
        return AIMessage(content="테스트 응답: OpenCV Canvas 마스킹 및 Redshift DW 아키텍처 기반입니다.")


class TestElasticsearchRAGPipeline(unittest.TestCase):
    def setUp(self):
        re_es._QUERY_EMBEDDING_CACHE.clear()
        self.chunks = re_es.load_and_split_documents(force_reload=True)

    def test_load_and_split_documents(self):
        self.assertGreater(len(self.chunks), 10)
        sources = {c.metadata["source"] for c in self.chunks}
        self.assertEqual(
            sources,
            {"PROFILE.md", "RAG_DASHBOARD.md", "SMART_RESTAURANT.md", "TOYSTORE_MOBILE_REVENUE.md"},
        )

    def test_build_index_body_nori_and_fallback(self):
        nori_body = re_es.build_es_index_body(embedding_dims=3072, use_nori=True)
        self.assertEqual(
            nori_body["settings"]["analysis"]["tokenizer"]["korean_nori_tokenizer"]["type"],
            "nori_tokenizer",
        )
        self.assertEqual(nori_body["mappings"]["properties"]["embedding"]["dims"], 3072)
        self.assertEqual(nori_body["mappings"]["properties"]["embedding"]["similarity"], "cosine")

        fallback_body = re_es.build_es_index_body(embedding_dims=768, use_nori=False)
        self.assertIn(
            "cjk_bigram",
            fallback_body["settings"]["analysis"]["analyzer"]["korean_analyzer"]["filter"],
        )
        self.assertEqual(fallback_body["mappings"]["properties"]["embedding"]["dims"], 768)

    def test_knn_and_bm25_query_dsl(self):
        plan = re_es.run_query_planner("스마트 맛집 추천 플랫폼에서 OpenCV Canvas 혼잡도 수치화 원리를 알려줘")
        self.assertEqual(plan["planner_mode"], "fast_rule_planner")
        self.assertIn("SMART_RESTAURANT.md", plan["target_sources"])

        knn_body = re_es.build_es_knn_request(
            query_vector=[0.1] * 8,
            top_k=10,
            target_sources=plan["target_sources"],
            category_filter=plan["category_filter"],
        )
        self.assertEqual(knn_body["knn"]["field"], "embedding")
        self.assertEqual(knn_body["knn"]["k"], 10)
        self.assertIn("filter", knn_body["knn"])

        bm25_body = re_es.build_es_bm25_request(query_plan=plan, top_k=10)
        self.assertIn("bool", bm25_body["query"])
        self.assertEqual(bm25_body["_source"]["excludes"], ["embedding"])

    def test_end_to_end_langgraph_with_mock_es(self):
        fake_client = MagicMock()
        restaurant_chunks = [c for c in self.chunks if c.metadata["source"] == "SMART_RESTAURANT.md"][:5]

        def _make_hit(chunk, score):
            return {
                "_id": chunk.metadata["chunk_id"],
                "_score": score,
                "_source": {
                    "chunk_id": chunk.metadata["chunk_id"],
                    "source": chunk.metadata["source"],
                    "doc_title": chunk.metadata["doc_title"],
                    "category": chunk.metadata["category"],
                    "section": chunk.metadata["section"],
                    "header_1": chunk.metadata.get("Header 1", ""),
                    "header_2": chunk.metadata.get("Header 2", ""),
                    "header_3": chunk.metadata.get("Header 3", ""),
                    "page_content": chunk.page_content,
                },
            }

        def _mock_search(index, body):
            if "knn" in body:
                hits = [_make_hit(c, 0.95 - i * 0.05) for i, c in enumerate(restaurant_chunks)]
            else:
                hits = [_make_hit(c, 12.0 - i * 1.1) for i, c in enumerate(reversed(restaurant_chunks))]
            return {"hits": {"hits": hits}}

        fake_client.search.side_effect = _mock_search
        fake_client.count.return_value = {"count": len(self.chunks)}

        fake_emb = _FakeEmbeddings(dims=8)
        es_store = re_es.ElasticsearchHybridStore(
            client=fake_client,
            embeddings=fake_emb,
            index_name="portfolio_rag_index",
            analyzer_mode="nori",
            embedding_dims=8,
            search_kwargs={"k": 4},
        )
        fake_llm = _FakeLLM()
        graph = re_es.build_langgraph_app(fake_llm, es_store)

        # 1) RAG 문서 질의 경로 검증
        answer, node, docs = re_es.run_query(
            graph=graph,
            llm=fake_llm,
            es_store=es_store,
            user_query="스마트 맛집 추천 플랫폼에서 OpenCV Canvas 혼잡도 수치화 원리를 알려줘",
            thread_id="test_thread_1",
            mode="auto",
        )
        self.assertEqual(node, "rag_bot")
        self.assertIn("OpenCV", answer)
        self.assertGreaterEqual(len(docs), 4)
        self.assertEqual(fake_emb.query_calls, 1)  # 임베딩 1회 호출 확인
        self.assertEqual(docs[0].metadata["query_plan_summary"]["es_analyzer"], "nori")
        self.assertIn("rrf_score", docs[0].metadata)
        self.assertIn("rerank_score", docs[0].metadata)

        # 2) 평가/피드백 chat_bot 경로 검증
        eval_ans, eval_node, eval_docs = re_es.run_query(
            graph=graph,
            llm=fake_llm,
            es_store=es_store,
            user_query="평가: Elasticsearch 검색 결과가 정확해서 좋았습니다.",
            thread_id="test_thread_1",
            mode="auto",
        )
        self.assertIn("평가", eval_node)
        self.assertEqual(eval_docs, [])


if __name__ == "__main__":
    unittest.main()
