import logging

logger = logging.getLogger("Retriever")

class HybridRetriever:
    """
    为了展示混合检索(Hybrid Search)和 Small2Big 层次化检索能力。
    面试话术：
        "老师，虽然向量检索(稠密 Vector)对语义相似度支持得好，但是它对'特定的专有缩写/编号(1234AB)'不敏感。
        因此我参考人大之前论文里提到的思路，把单纯的 Embedding 升级成了 Hybrid Search (混合检索)。
        我利用元数据(Metadata)在搜索前做了一次倒排索引式的强过滤。
        同时，在切分文档时，我参考了 Small2Big 策略：用很短的一句话(Proposition)去匹配问题，
        一旦命中，就顺藤摸瓜返回它所属的非常完整的一大段话被处理的模型，避免了断章取义碎成渣。"
    """
    def __init__(self, backend_store):
        self.store = backend_store

    def retrieve(self, query: str, filters: dict = None):
        logger.info(f"Starting Hybrid Retrieval for: '{query}'")

        # 1. 元数据过滤阶段 (Meta Filtering)
        logger.info(f"Step 1: Applying metadata filters: {filters}")

        # 2. 向量检索阶段 (Vector Search via real LanceDB + bge-m3)
        logger.info("Step 2: Performing real Vector Search via SiliconFlow bge-m3 + LanceDB ANN...")
        results = self.store.search(query, top_k=3, filters=filters)

        if not results:
            logger.warning("No results found from vector store.")
            return None

        # 取分数最高的结果（LanceDB 按距离排序，_distance 越小越相关）
        best = results[0]
        hit = {
            "doc_id": str(best.get("chunk_id", "unknown")),
            "chunk_text": best.get("content", ""),
            "score": 1.0 - float(best.get("score", 0.5)),  # 将距离转换为相似度分数
            "metadata": {"year": best.get("metadata_year", 2024), "type": "paragraph"},
        }

        # 3. Small2Big 扩展阶段：优先使用检索结果中的完整父块内容
        logger.info(f"Step 3: Small2Big expansion. Hit chunk: '{hit['chunk_text'][:30]}...'. Expanding to full context...")
        
        # 优先取分数最高那个块的 parent_content
        parent_context = best.get("parent_content")
        if parent_context:
            hit["expanded_context"] = parent_context
            logger.info("Successfully expanded to dedicated Parent Content.")
        else:
            # 兜底方案：将所有 Top-K 检索结果拼接作为上下文
            extended_texts = [r.get("content", "") for r in results]
            hit["expanded_context"] = "\n".join(extended_texts)
            logger.info("No dedicated Parent Content found, using Top-K concatenation.")

        logger.info(f"Retrieved Result: Score={hit['score']:.4f}, DocID={hit['doc_id']}")
        return hit


if __name__ == "__main__":
    from storage import DistributedVectorStore
    store = DistributedVectorStore(1)
    retriever = HybridRetriever(store)

    # 正常查询带年份过滤
    retriever.retrieve("人大信院的优势在哪里？", filters={"year": 2024})
