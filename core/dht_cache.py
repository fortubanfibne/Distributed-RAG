import logging
import hashlib
from typing import Optional, Dict

logger = logging.getLogger("DHT-Cache")

class DHTCacheLayer:
    """
    为了解决检索慢、大批量相似请求击穿向量库的问题而设计的。
    面试话术：
        "老师，RAG 的通病是大家问的问题都差不多（比如财报多少钱）。
        每次都去跑一遍百亿参数的 Embedding 再去库里全表扫描（即使是 ANN 也慢），这在生产环境不可接受。
        我借鉴了 Memcached/Redis 的设计，在 Rerank (重排) 阶段之前加了一个基于 DHT（分布式哈希表）概念的 Cache 层。"
    """
    def __init__(self):
        # 模拟内存中的分布式键值存储
        self.cache: Dict[str, dict] = {}
        
    def _generate_key(self, query: str, filters: Optional[dict] = None) -> str:
        """
        生成缓存的唯一 Key。
        关键点：连同 Metadata(如年份) 一起 Hash，保证“2023年财报”和“2024年财报”不冲突。
        """
        key_str = query
        if filters:
            # 排序后组合，保证相同的条件生成相同的字符串
            sorted_filters = dict(sorted(filters.items()))
            key_str += f"|_filters:{str(sorted_filters)}"
            
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
    def get(self, query: str, filters: Optional[dict] = None) -> Optional[dict]:
        """尝试从 Cache 命中"""
        key = self._generate_key(query, filters)
        if key in self.cache:
            logger.info(f"[HIT] Cache hit for query: '{query}'")
            return self.cache[key]
        logger.info(f"[MISS] Cache miss for query: '{query}'")
        return None
        
    def set(self, query: str, filters: dict, result: dict):
        """写入 Cache"""
        key = self._generate_key(query, filters)
        self.cache[key] = result
        logger.info(f"[SET] Cached result for '{query}'. Total cache keys: {len(self.cache)}")
        
if __name__ == "__main__":
    cache = DHTCacheLayer()
    q = "核心团队都有谁？"
    f = {"year": 2024}
    
    # 第一次查询 (必定 Miss)
    res = cache.get(q, f)
    if not res:
        # 假设去底层的向量库查到了结果
        db_result = {"answer": "孟老师、杜老师等", "score": 0.99}
        cache.set(q, f, db_result)
        
    # 第二次相同的查询 (命中 Hit)
    cache.get(q, f)
    
    # 第三次条件不同的查询 (Miss)
    cache.get(q, {"year": 2023})
