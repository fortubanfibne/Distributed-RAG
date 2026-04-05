import time
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Raft")

# 全局写锁：保护多线程并发写入 success_count 时的竞态条件
_count_lock = threading.Lock()

class RaftNode:
    """
    模拟 Raft 协议中的单个节点。
    在真实面试中，你可以这样解释这个类：
    “为了保证我的向量索引高可用，我用 Python 线程模拟了 6.5840 里的 Raft 节点。
    当新的 PDF 写入时，Leader 节点会通过 AppendEntries (WAL 日志追赶) 把增量索引同步给 Follower，
    从而解决传统大模型知识库容易丢失数据的问题。”
    """
    def __init__(self, node_id, is_leader=False):
        self.node_id = node_id
        self.state = "LEADER" if is_leader else "FOLLOWER"
        self.log = []  # 模拟 Write-Ahead Log (WAL)
        self.term = 1
        self._log_lock = threading.Lock()  # 保护本地日志的并发写入

    def append_entries(self, entries, leader_term: int = 1):
        """
        模拟 Leader 将新 Chunk 日志追加给 Follower。
        加入 term 校验：若 Follower 的 term 更高，说明当前 Leader 已过期（stale leader），
        拒绝写入。这是 Raft 论文 §5.1 的核心安全保证。
        """
        # Raft term 验证：拒绝过期 Leader 的请求
        if leader_term < self.term:
            logger.warning(
                f"Node {self.node_id} rejected AppendEntries: "
                f"leader_term={leader_term} < my_term={self.term} (stale leader)"
            )
            return False

        if self.state == "FOLLOWER":
            time.sleep(0.1)  # 模拟网络 RTT
            with self._log_lock:  # 保护并发写入本地日志
                self.log.extend(entries)
            logger.info(
                f"Node {self.node_id} (Follower) appended {len(entries)} entries. "
                f"Total WAL size: {len(self.log)}"
            )
            return True
        return False

class RaftCluster:
    def __init__(self, node_count=3):
        self.nodes = [RaftNode(i, is_leader=(i==0)) for i in range(node_count)]
        self.leader = self.nodes[0]
        
    def write_data(self, data):
        """
        模拟客户端写入数据（例如一条被切分好的向量）。
        触发强一致性同步：Leader 先写本地 WAL，再并发广播给所有 Follower，
        超过半数（Quorum）确认后才视为提交成功。
        """
        logger.info(f"Leader received write request: {data[:20]}...")

        # 1. Leader 写入本地 WAL
        self.leader.log.append(data)

        # 2. 并发广播给所有 Follower
        #    ⚠️  success_count 被多个线程并发读写，必须用 Lock 保护，否则是竞态条件
        success_count = [1]  # Leader 自己算一票，使用列表以便在闭包中修改
        lock = threading.Lock()

        def send_to_follower(node):
            if node.append_entries([data], leader_term=self.leader.term):
                with lock:  # 临界区：原子地让计数器 +1
                    success_count[0] += 1

        threads = []
        followers = self.nodes[1:]
        for node in followers:
            t = threading.Thread(target=send_to_follower, args=(node,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # 3. 多数派（Quorum）确认
        quorum = len(self.nodes) // 2 + 1
        if success_count[0] >= quorum:
            logger.info(f"Quorum reached ({success_count[0]}/{len(self.nodes)}). WAL committed successfully.")
            return True
        else:
            logger.error(f"Failed to reach quorum ({success_count[0]}/{len(self.nodes)} < {quorum}).")
            return False

if __name__ == "__main__":
    # 测试代码
    cluster = RaftCluster()
    cluster.write_data("document_chunk_1_vector_data")
