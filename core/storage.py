import logging
import hashlib
import bisect
from .raft import RaftCluster

logger = logging.getLogger("Storage")


class ConsistentHashRing:
    """
    真正的一致性哈希环实现。
    ─────────────────────────────────────────────────────
    核心思想：
    1. 把 [0, 2^32) 的整数空间首尾相连，构成一个"环"。
    2. 每个物理节点（shard）在环上放 virtual_nodes 个虚拟节点，
       让节点均匀散落在环上，避免数据倾斜。
    3. 路由时：把 doc_id 哈希成一个整数，在有序列表上
       用 bisect 做 O(log n) 二分查找，顺时针找到的
       第一个虚拟节点就是目标 shard。
    ─────────────────────────────────────────────────────
    优势对比简单取模（hash % n）:
    - 扩容时只需迁移 1/n 的数据，而非全量重洗
    - 节点宕机只影响相邻节点，不引发全局雪崩
    """

    def __init__(self, virtual_nodes: int = 100):
        self.virtual_nodes = virtual_nodes
        # 有序列表：存放所有虚拟节点的哈希值（用于二分查找）
        self._ring_keys: list[int] = []
        # 哈希值 → 物理 shard_id 的映射
        self._ring_map: dict[int, int] = {}

    def _hash(self, key: str) -> int:
        """把任意字符串映射到 [0, 2^32) 上的一个整数"""
        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % (2**32)

    def add_node(self, shard_id: int):
        """向哈希环中添加一个物理节点（同时撒入虚拟节点）"""
        for i in range(self.virtual_nodes):
            vnode_key = f"shard-{shard_id}#vnode-{i}"
            h = self._hash(vnode_key)
            self._ring_map[h] = shard_id
            bisect.insort(self._ring_keys, h)  # 保持有序
        logger.debug(f"Node shard-{shard_id} added to ring with {self.virtual_nodes} virtual nodes.")

    def remove_node(self, shard_id: int):
        """从哈希环中移除一个物理节点（摘除所有虚拟节点）"""
        for i in range(self.virtual_nodes):
            vnode_key = f"shard-{shard_id}#vnode-{i}"
            h = self._hash(vnode_key)
            if h in self._ring_map:
                self._ring_map.pop(h, None)
                idx = bisect.bisect_left(self._ring_keys, h)
                if idx < len(self._ring_keys) and self._ring_keys[idx] == h:
                    self._ring_keys.pop(idx)
        logger.debug(f"Node shard-{shard_id} removed from ring.")

    def get_node(self, doc_id: str) -> int:
        """
        路由函数：给定 doc_id，返回负责它的 shard_id。
        - 把 doc_id 哈希到环上
        - bisect_right 找到顺时针方向第一个虚拟节点
        - 若超过末尾则回绕到环头（取模），实现"环"语义
        """
        if not self._ring_keys:
            raise RuntimeError("Hash ring is empty — no nodes registered.")
        h = self._hash(doc_id)
        idx = bisect.bisect_right(self._ring_keys, h) % len(self._ring_keys)
        return self._ring_map[self._ring_keys[idx]]


class DistributedVectorStore:
    """
    基于真正一致性哈希环的分布式向量存储。
    ─────────────────────────────────────────────────────
    面试话术：
    "老师，这里用 ConsistentHashRing 替换了之前简单的
     hash % n 取模。通过哈希环 + bisect 二分查找，每个
     物理节点挂 100 个虚拟节点散列到 [0, 2^32) 的环上。
     新文档 MD5 后在有序列表二分找到顺时针第一个虚拟节点
     即为目标分片。这样横向扩容（add_node）时只需迁移
     约 1/n 的数据，不会引发全局缓存雪崩。"
    ─────────────────────────────────────────────────────
    """

    def __init__(self, num_shards: int = 3, virtual_nodes: int = 100):
        self.num_shards = num_shards
        # 构建一致性哈希环
        self.ring = ConsistentHashRing(virtual_nodes=virtual_nodes)
        # 每个 shard 下面挂一个包含 3 副本的 Raft 集群
        self.shards: dict[int, RaftCluster] = {}
        for i in range(num_shards):
            self.shards[i] = RaftCluster(3)
            self.ring.add_node(i)
        logger.info(
            f"DistributedVectorStore initialized: "
            f"{num_shards} shards × {virtual_nodes} virtual nodes on consistent hash ring."
        )

    def add_shard(self, shard_id: int):
        """
        动态横向扩容：新增一个 shard。
        由于一致性哈希，只有约 1/n 的 key 需要迁移。
        """
        if shard_id in self.shards:
            logger.warning(f"Shard-{shard_id} already exists, skip.")
            return
        self.shards[shard_id] = RaftCluster(3)
        self.ring.add_node(shard_id)
        logger.info(f"Shard-{shard_id} added to cluster with minimal data migration.")

    def remove_shard(self, shard_id: int):
        """动态缩容：摘除一个 shard（生产环境需先迁移数据）"""
        if shard_id not in self.shards:
            return
        self.ring.remove_node(shard_id)
        self.shards.pop(shard_id, None)
        logger.info(f"Shard-{shard_id} removed from cluster.")

    def _get_shard_id(self, doc_id: str) -> int:
        """通过一致性哈希环路由到目标 shard"""
        return self.ring.get_node(doc_id)

    def insert(self, chunk: dict):
        """
        写入一个向量 Chunk，由一致性哈希决定落哪个 shard，
        再通过底层 Raft 做多副本强一致性持久化。
        chunk: {"doc_id": "pdf-123", "text": "...", "vector": [0.1, 0.2]}
        """
        doc_id = chunk.get("doc_id", "unknown")
        shard_id = self._get_shard_id(doc_id)

        logger.info("-" * 40)
        logger.info(f"[ConsistentHash] Routing doc '{doc_id}' → Shard-{shard_id}")

        raft_cluster = self.shards[shard_id]
        success = raft_cluster.write_data(f"DATA: {chunk['text'][:15]}...")

        if success:
            logger.info(f"Chunk successfully persisted in Shard-{shard_id}")
        else:
            logger.error(f"Failed to persist chunk in Shard-{shard_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = DistributedVectorStore(num_shards=3)
    store.insert({"doc_id": "math_paper_01", "text": "Linear Algebra..."})
    store.insert({"doc_id": "cs_paper_02", "text": "Distributed Systems..."})

    # 演示动态扩容：新增 shard-3，只迁移约 1/4 的数据
    print("\n=== 动态扩容：add shard-3 ===")
    store.add_shard(3)
    store.insert({"doc_id": "ml_paper_03", "text": "Transformer & Attention..."})
