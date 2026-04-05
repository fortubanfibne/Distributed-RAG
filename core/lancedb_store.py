import logging
import time
import os
from typing import List, Dict, Any
import lancedb
import pyarrow as pa
import numpy as np
import httpx
import json

# ========== 硅基流动 API 配置 ==========
SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-lthsvztszxotjlcxocgfgitztcpxtcnhkwklzuejbbfstozd"
)
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # bge-m3 输出维度为 1024


def get_embedding(text: str) -> List[float]:
    """调用硅基流动 BAAI/bge-m3 模型获取真实文本向量"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
        "encoding_format": "float",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{SILICONFLOW_BASE_URL}/embeddings",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

logger = logging.getLogger("LanceDB-Store")

class LanceDBStore:
    def __init__(self, db_path: str = "./lancedb_data"):
        """
        LanceDB 轻量级嵌入式存储后端
        
        面试话术：
        "老师，在工程实现上，我设计了基于 LanceDB 的存储策略。
        LanceDB 是基于 Rust 和 Apache Arrow 生态构建的无服务器(Serverless)向量数据库。
        它直接内嵌在进程中，没有任何 RPC 调用的网络折损。
        更重要的是，它默认采用基于磁盘的索引(Disk-based Index)，
        在处理中等规模的分块文档时，能在不挤占宝贵 RAM 的情况下实现极速检索，
        这是我脱离单纯的『API 调用』，向『存储引擎深度优化』迈出的务实一步。"
        """
        logger.info(f"Initializing LanceDB embedded store at {db_path}...")
        self.db = lancedb.connect(db_path)
        self.table_name = "dist_rag_docs"
        
        # Define the schema if table doesn't exist
        # id: str, vector: List[float], text: str, meta_year: int
        if self.table_name not in self.db.table_names():
            logger.info("Creating new LanceDB table with Small2Big support...")
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),  # bge-m3: 1024 dim
                pa.field("text", pa.string()),          # 子分块文本（用于显示）
                pa.field("parent_text", pa.string()),   # 父分块文本（用于 Small2Big 召回）
                pa.field("meta_year", pa.int32()),
            ])
            self.table = self.db.create_table(self.table_name, schema=schema)
        else:
            logger.info("Connecting to existing LanceDB table...")
            self.table = self.db.open_table(self.table_name)
            
    def insert(self, chunk: Dict[str, Any]):
        """真实向量插入：调用 BAAI/bge-m3 生成语义向量后存入 LanceDB"""
        logger.info(f"    [LanceDB] Inserting Chunk ID={chunk['chunk_id']} — Calling SiliconFlow bge-m3 for embedding...")

        # 真实 Embedding：调用硅基流动 BAAI/bge-m3
        text = chunk.get('content') or chunk.get('text', '')
        vector = get_embedding(text)

        data = [{
            "id": str(chunk['chunk_id']),
            "vector": vector,
            "text": text,
            "parent_text": chunk.get('parent_content', text), # 如果没有父分块，则默认使用自身
            "meta_year": chunk.get('metadata', {}).get('year', 2024)
        }]

        self.table.add(data)
        logger.info(f"    [LanceDB] Insert complete. Real semantic vector stored (dim={len(vector)}).")
            
    def search(self, query: str, top_k: int = 3, filters: dict = None) -> List[Dict[str, Any]]:
        """真实混合检索：用 bge-m3 向量化 query，再在 LanceDB 中做 ANN 向量检索"""
        logger.info(f"    [LanceDB] Executing Hybrid Search for query: '{query}'")

        # 真实 Query Embedding
        query_vector = get_embedding(query)
        logger.info(f"    [LanceDB] Query embedded (dim={len(query_vector)}). Running ANN search...")

        search_obj = self.table.search(query_vector).limit(top_k)
        
        if filters and "year" in filters:
            logger.info(f"    [LanceDB] Applying Pre-filtering metadata: year={filters['year']}")
            search_obj = search_obj.where(f"meta_year = {filters['year']}", prefilter=True)
            
        results = search_obj.to_list()
        
        # 组装返回格式，保持与上层业务兼容
        docs = []
        for r in results:
             docs.append({
                 "chunk_id": r["id"],
                 "content": r["text"],
                 "parent_content": r.get("parent_text", r["text"]),
                 "metadata_year": r["meta_year"],
                 "score": r["_distance"]
             })
             
        logger.info(f"    [LanceDB] Found {len(docs)} segments directly from disk index.")
        return docs
