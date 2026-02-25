"""
End-to-end query pipeline — orchestrates retrieval → generation → logging.
This is the main entry point for processing citizen queries.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.utils.config import AppConfig, load_config
from src.utils.text_processing import (
    preprocess_query, token_count, classify_topic,
    compute_sentiment, compute_readability, count_citations,
    count_hedge_words, detect_refusal,
)
from src.production.retrieval import RetrievalPipeline, RetrievalResult
from src.production.generation import ResponseGenerator, GenerationResult
from src.production.logging import LogStore, ProductionLog, generate_id, utcnow


class QueryPipeline:
    """
    Full production query pipeline:
    1. Query intake + preprocessing
    2. Retrieval (Pinecone integrated embedding)
    3. LLM generation (Groq via litellm)
    4. Descriptor extraction
    5. Production logging (SQLite)
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        log_store: Optional[LogStore] = None,
        retrieval: Optional[RetrievalPipeline] = None,
        generator: Optional[ResponseGenerator] = None,
        session_id: str = "",
    ):
        self.config = config or load_config()
        self.log_store = log_store or LogStore(self.config.db_path)
        self.retrieval = retrieval or RetrievalPipeline(self.config.retrieval)
        self.generator = generator or ResponseGenerator(self.config.primary_llm)
        self.session_id = session_id

    def process(self, query: str) -> ProductionLog:
        """
        Process a single user query end-to-end.
        Returns the full ProductionLog (also written to SQLite).
        """
        query_id = generate_id()
        timestamp = utcnow()

        # 1. Preprocess
        cleaned = preprocess_query(query)
        topic = classify_topic(cleaned)
        q_tokens = token_count(cleaned)

        # 2. Retrieve
        retrieval_result = self.retrieval.retrieve(cleaned)

        # 3. Generate
        generation_result = self.generator.generate(cleaned, retrieval_result)

        # 4. Extract descriptors
        raw_resp = generation_result.raw_response
        resp_length = len(raw_resp.split())
        sentiment = compute_sentiment(raw_resp)
        readability = compute_readability(raw_resp)
        citations = count_citations(raw_resp)
        hedges = count_hedge_words(raw_resp)
        is_refusal = detect_refusal(raw_resp)

        # Context token ratio
        ctx_ratio = (
            retrieval_result.context_token_count / max(generation_result.completion_tokens, 1)
        )

        # 5. Build log
        log = ProductionLog(
            query_id=query_id,
            timestamp=timestamp,
            raw_query=query,
            cleaned_query=cleaned,
            query_topic=topic,
            query_token_count=q_tokens,
            top_k_chunk_ids=retrieval_result.chunk_ids,
            top_k_scores=retrieval_result.scores,
            reranked_chunk_ids=retrieval_result.reranked_ids,
            selected_context_ids=retrieval_result.selected_ids,
            context_token_count=retrieval_result.context_token_count,
            model_name=generation_result.model_name,
            temperature=generation_result.temperature,
            prompt_tokens=generation_result.prompt_tokens,
            completion_tokens=generation_result.completion_tokens,
            raw_response=raw_resp,
            final_response=generation_result.final_response,
            finish_reason=generation_result.finish_reason,
            response_length=resp_length,
            response_sentiment=sentiment,
            response_readability=readability,
            citation_count=citations,
            hedge_word_count=hedges,
            refusal_flag=is_refusal,
            session_id=self.session_id,
            response_latency_ms=generation_result.latency_ms,
            mean_retrieval_distance=retrieval_result.mean_retrieval_distance,
            context_token_ratio=ctx_ratio,
        )

        # 6. Write to database
        self.log_store.write(log)

        return log
