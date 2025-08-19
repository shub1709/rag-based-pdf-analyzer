import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines dense (vector DB) and sparse (BM25) retrieval with reranking"""

    def __init__(self, embeddings_manager, cross_encoder_model: str = None):
        self.embeddings_manager = embeddings_manager
        self.cross_encoder_model = CrossEncoder(cross_encoder_model) if cross_encoder_model else None
        self.bm25_corpus = []
        self.bm25 = None

        if cross_encoder_model:
            self.cross_encoder_model = CrossEncoder(cross_encoder_model)
            logger.info(f"[HybridRetriever] Loaded CrossEncoder reranker: {cross_encoder_model}")
        else:
            self.cross_encoder_model = None
            logger.warning("[HybridRetriever] No reranker configured — skipping reranking")


    def build_bm25_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from corpus (for sparse retrieval)"""
        texts = [doc["content"] for doc in documents]
        tokenized = [t.split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_corpus = documents
        logger.info("BM25 index built with %d documents", len(texts))

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Perform hybrid search with fallback + reranking"""
        results = []

        # --- Dense retrieval (Chroma) ---
        dense_results = []
        try:
            dense_results = self.embeddings_manager.similarity_search(query, k=k)
        except Exception as e:
            logger.warning(f"Dense retrieval failed: {e}")

        results.extend(dense_results)

        # --- Sparse retrieval (BM25) ---
        bm25_results = []
        if self.bm25:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            bm25_results = [self.bm25_corpus[i] for i in top_indices]
            for r, score in zip(bm25_results, [scores[i] for i in top_indices]):
                r["bm25_score"] = float(score)

        results.extend(bm25_results)

        # --- Fallback logic ---
        if not results:  # if both fail
            logger.warning("No results from dense or sparse retrieval.")
            return []

        if len(results) < k and bm25_results:  
            # If dense returned too few, fill from BM25
            results.extend(bm25_results)

        if len(results) < k and dense_results:  
            # If sparse returned too few, fill from dense
            results.extend(dense_results)

        # Deduplicate by content
        seen = set()
        unique_results = []
        for r in results:
            if r["content"] not in seen:
                unique_results.append(r)
                seen.add(r["content"])

        # --- Rerank with cross-encoder ---
        if self.cross_encoder_model and unique_results:
            pairs = [(query, r["content"]) for r in unique_results]
            try:
                scores = self.cross_encoder_model.predict(pairs)
                for r, s in zip(unique_results, scores):
                    r["rerank_score"] = float(s)
                unique_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        print("\n[RERANKING DEBUG]")
        for r in unique_results[:5]:
            print(f"Chunk: {r['content'][:80]}... | Score: {r.get('rerank_score', 'N/A')}")


        return unique_results[:k]

