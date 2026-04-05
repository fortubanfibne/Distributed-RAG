import os
import sys
import shutil

# Get the directory of the current script (Dist-RAG)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the Dist-RAG directory to sys.path if it's not already there
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import logging
import time
from typing import Optional

from core.document_parser import DeepDocumentParser
from core.storage import DistributedVectorStore
from core.lancedb_store import LanceDBStore
from core.dht_cache import DHTCacheLayer
from core.retriever import HybridRetriever
from core.inference import DistributedInferenceEngine
from core.evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Dist-RAG-Main")

# ================================================================== #
#  论文下载配置：5 篇经典 CS 论文作为真实知识库数据源
# ================================================================== #
PAPERS = {
    "raft.pdf": "https://raft.github.io/raft.pdf",
    "mapreduce.pdf": "https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf",
    "gfs.pdf": "https://static.googleusercontent.com/media/research.google.com/en//archive/gfs-sosp2003.pdf",
    "attention.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
}

DATA_DIR = os.path.join(current_dir, "data")


def download_papers():
    """
    自动下载 5 篇经典 CS 论文到 data/ 目录。
    面试话术：
        "我的知识库使用了 5 篇计算机领域最经典的论文作为真实数据源——
        Raft 共识算法、Google MapReduce、GFS 分布式文件系统、
        Transformer 原论文 Attention Is All You Need、以及 BERT。
        这些论文与我的项目架构直接对应：Raft 对应存储层共识协议，
        MapReduce 对应 ETL 并发架构，Attention/BERT 对应 Embedding 模型基础。"
    """
    import httpx

    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"📥 Checking/downloading {len(PAPERS)} papers to '{DATA_DIR}'...")

    for filename, url in PAPERS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
            logger.info(f"  ✅ '{filename}' already exists, skipping.")
            continue

        logger.info(f"  ⬇️  Downloading '{filename}' from {url}...")
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(resp.content)
            logger.info(f"  ✅ '{filename}' downloaded ({len(resp.content) / 1024:.0f} KB)")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to download '{filename}': {e}")

    downloaded = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    logger.info(f"📚 Data directory ready: {len(downloaded)} PDFs available.")
    return len(downloaded)


class DistRAGSystem:
    def __init__(self, use_lancedb: bool = False):
        """
        use_lancedb=False : 使用自研 Raft+Sharding Mock 后端（展示分布式原理）
        use_lancedb=True  : 使用 LanceDB 真实向量库（展示工程落地能力）
        面试话术：
            '老师，我将存储后端设计为可插拔的接口。
             开发演示阶段用自研的 Raft+Sharding Mock 展示分布式原理，
             一旦接入真实场景，只需将 use_lancedb=True，
             系统就会无缝切换到 LanceDB 嵌入式向量数据库，
             无需修改任何上层检索逻辑。这是系统设计中
             接口稳定、实现可替换的核心原则。'
        """
        mode_str = "LanceDB (Real Embedded DB)" if use_lancedb else "Raft+Sharding (Mock Distributed)"
        logger.info(f"Initializing Dist-RAG System | Storage Backend: {mode_str}")

        # 1. 亮点1：可插拔的向量存储后端
        if use_lancedb:
            # 真实 LanceDB 后端：无服务器开销，基于磁盘存储
            lancedb_db_path = os.path.join(current_dir, "dist_rag_lancedb")
            self.store = LanceDBStore(db_path=lancedb_db_path)
            logger.info("[Storage] Using LanceDB backend (Real Embedded Vector DB)")
        else:
            # 自研 Raft+Sharding Mock 后端：展示分布式原理
            self.store = DistributedVectorStore(num_shards=2)
            logger.info("[Storage] Using Raft+Sharding Mock backend (Distributed Simulation)")
        
        # 2. Rerank 前的分布式缓存层
        self.cache = DHTCacheLayer()
        
        # 3. 带元数据过滤和 Small2Big 的混合检索器
        self.retriever = HybridRetriever(self.store)
        
        # 4. 亮点2：基于 Map-Reduce 并发思想与 Small2Big 级联的真实 PDF 解析器
        self.parser = DeepDocumentParser(
            num_workers=4, 
            parent_chunk_size=1200, 
            child_chunk_size=400, 
            chunk_overlap=50
        )
        
        # 5. 亮点3：分布式推理引擎 (接入硅基流动 Qwen2.5-7B-Instruct，架构参考 vLLM 张量并行)
        self.inference_engine = DistributedInferenceEngine(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            tensor_parallel_size=4
        )
        
        # 6. RAG 质量评估器 (参考 DREAM/Ragas 评测体系)
        self.evaluator = RAGEvaluator()
        
    def upload_document(self, file_path: str):
        """文档上传与处理流水线（真实 PDF 解析）"""
        logger.info(f"========== STEP 1: PARALLEL DISTRIBUTED PARSING ==========")
        chunks = self.parser.parse_pdf(file_path)
        
        logger.info(f"========== STEP 2: DISTRIBUTED INGESTION (RAFT & SHARDING) ==========")
        for chunk in chunks:
            # 存入分布式向量库 (会触发路由和 Raft WAL 同步)
            self.store.insert(chunk)
            
        logger.info("Document ingested successfully.")
        return len(chunks)

    def upload_directory(self, dir_path: str):
        """批量上传目录下所有 PDF"""
        logger.info(f"========== BATCH INGESTION: '{dir_path}' ==========")
        chunks = self.parser.parse_directory(dir_path)

        logger.info(f"========== STEP 2: DISTRIBUTED INGESTION ({len(chunks)} chunks) ==========")
        for i, chunk in enumerate(chunks):
            self.store.insert(chunk)
            if (i + 1) % 20 == 0:
                logger.info(f"  ... ingested {i + 1}/{len(chunks)} chunks")

        logger.info(f"✅ Batch ingestion complete: {len(chunks)} chunks stored.")
        return len(chunks)

    def query(self, question: str, filters: Optional[dict] = None):
        """客户端发起检索请求"""
        logger.info(f"\n========== NEW QUERY: '{question}' ==========")
        
        # STEP 1: 尝试穿透缓存
        cached_result = self.cache.get(question, filters)
        if cached_result:
            logger.info("⚡ Returning instantly from DHT Cache! (Cache Avalanche Prevented)")
            return cached_result
            
        # STEP 2: 缓存未命中，执行重量级的混合检索
        logger.info("Cache miss. Fallback to heavy Hybrid Retrieval.")
        retrieved_docs = self.retriever.retrieve(question, filters)
        
        if not retrieved_docs:
            return "No relevant documents found."
            
        raw_context = retrieved_docs['expanded_context']
            
        # STEP 3: 调用分布式推理引擎 (亮点3：vLLM 张量并行 — 独立模块)
        logger.info("========== STEP 3: DISTRIBUTED INFERENCE (TENSOR PARALLELISM) ==========")
        final_answer = self.inference_engine.generate(
            context=raw_context,
            question=question
        )
        
        # STEP 4: 写入缓存供下次高频调用
        self.cache.set(question, filters, {"answer": final_answer, "source_docs": retrieved_docs['doc_id']})
        
        # STEP 5: Ragas 风格质量评估 (参考 DREAM 分布式实验框架的评测思路)
        logger.info("========== STEP 5: RAG QUALITY EVALUATION (Ragas-style) ==========")
        eval_report = self.evaluator.evaluate(
            question=question,
            answer=final_answer,
            context=raw_context
        )
        
        return {"answer": final_answer, "eval": eval_report}

if __name__ == "__main__":
    print("\n\n" + "="*60)
    print("🎓 Dist-RAG 分布式知识引擎 — 真实论文知识库演示")
    print("="*60 + "\n")

    # ============================================================ #
    #  STEP 0: 自动下载经典论文数据集
    # ============================================================ #
    print(">>> [0/4] 准备真实数据：下载经典 CS 论文 <<<")
    num_papers = download_papers()
    if num_papers == 0:
        print("❌ 没有可用的 PDF 文件，请检查网络连接。")
        sys.exit(1)
    time.sleep(0.5)

    # ============================================================ #
    #  STEP 1: 初始化系统 + 批量入库
    # ============================================================ #
    # 清除旧数据，确保干净的演示环境
    lancedb_path = os.path.join(current_dir, "dist_rag_lancedb")
    if os.path.exists(lancedb_path):
        shutil.rmtree(lancedb_path)
        logger.info("🗑️  Cleared old LanceDB data for fresh demo.")

    rag = DistRAGSystem(use_lancedb=True)
    time.sleep(0.5)

    print("\n>>> [1/4] 批量解析 & 分布式入库 (真实 PDF + Map-Reduce 并发 + Raft 同步) <<<")
    total_chunks = rag.upload_directory(DATA_DIR)
    print(f"\n📊 入库统计: {num_papers} 篇论文 → {total_chunks} chunks 已分布式存储\n")
    time.sleep(1)

    # ============================================================ #
    #  STEP 2: 第一次查询（缓存未命中，走完整链路）
    # ============================================================ #
    print(">>> [2/4] 演示完整查询链路 (混合检索 → 分布式推理 → LLM 评估) <<<")
    q1 = "How does Raft handle leader election when the current leader fails?"
    res1 = rag.query(q1)
    if isinstance(res1, dict):
        print(f"\n=> 查询: {q1}")
        print(f"=> 回答: {res1['answer']}")
        ev = res1['eval']
        print(f"   📊 Ragas 评估 | Faithfulness={ev['faithfulness']:.3f}  "
              f"Relevancy={ev['answer_relevancy']:.3f}  "
              f"Precision={ev['context_precision']:.3f}  "
              f"综合={ev['composite_score']:.3f}\n")
    else:
        print(f"\n=> 回答: {res1}\n")

    time.sleep(1)

    # ============================================================ #
    #  STEP 3: 同一查询第二次（演示 DHT 缓存命中）
    # ============================================================ #
    print(">>> [3/4] 演示 DHT 缓存命中 (相同查询瞬间返回) <<<")
    res2 = rag.query(q1)
    if isinstance(res2, dict):
        print(f"=> 回答(Cache Hit): {res2['answer']}\n")
    else:
        print(f"=> 回答: {res2}\n")

    time.sleep(1)

    # ============================================================ #
    #  STEP 4: 跨论文查询（展示多文档知识融合）
    # ============================================================ #
    print(">>> [4/4] 演示跨论文知识检索 (Attention 论文) <<<")
    q2 = "What is the self-attention mechanism and how does it compute query, key and value?"
    res3 = rag.query(q2)
    if isinstance(res3, dict):
        print(f"\n=> 查询: {q2}")
        print(f"=> 回答: {res3['answer']}")
        ev = res3['eval']
        print(f"   📊 Ragas 评估 | Faithfulness={ev['faithfulness']:.3f}  "
              f"Relevancy={ev['answer_relevancy']:.3f}  "
              f"Precision={ev['context_precision']:.3f}  "
              f"综合={ev['composite_score']:.3f}\n")
    else:
        print(f"\n=> 回答: {res3}\n")

    print("\n✅ 系统展示完毕。谢谢老师！祝复试高分！🎉")
