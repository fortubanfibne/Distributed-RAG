import logging
import re
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pdfplumber

logger = logging.getLogger("Dist-Parser")


class DeepDocumentParser:
    """
    亮点2：真实 PDF 解析 + 并行化预处理（参考 Spark/Ray Map-Reduce 思想）

    面试话术：
        "老师，在真实的企业级 RAG 中，我们面对的不是几篇 PDF，而是百万级文档的存量清洗。
        传统的单线程解析和 Embedding 是灾难级的慢。
        在我的架构中，我抛弃了单机串行方案，引入了类似于 Spark/Ray 的分布式 Map-Reduce 思想：
        Map 阶段用 ThreadPoolExecutor 将文档按页粒度并发下发给多个 Worker 做文本提取和清洗；
        Reduce 阶段统一调用 API 做 bge-m3 向量化并写入磁盘。
        我使用 pdfplumber 做真实的 PDF 版面解析，并实现了递归字符分割 + 滑动窗口 Overlap 的分块策略，
        确保每个 chunk 既有高语义密度，又不丢失上下文连贯性。"
    """

    # 递归分割的分隔符优先级（从粗到细）
    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, num_workers: int = 4, parent_chunk_size: int = 1200, child_chunk_size: int = 400, chunk_overlap: int = 50):
        """
        Args:
            num_workers: 并发 Worker 数量
            parent_chunk_size: 父分块目标长度（提供给 LLM 的上下文）
            child_chunk_size: 子分块目标长度（用于 Embedding 检索）
            chunk_overlap: 子分块之间的重叠字符数
        """
        self.num_workers = num_workers
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized Small2Big Parser | workers={self.num_workers}, "
            f"parent_sz={self.parent_chunk_size}, child_sz={self.child_chunk_size}, overlap={self.chunk_overlap}"
        )

    # ------------------------------------------------------------------ #
    #  核心：递归字符分割 + 滑动窗口 Overlap
    # ------------------------------------------------------------------ #

    def _recursive_split(self, text: str, chunk_size: int, separators: Optional[List[str]] = None) -> List[str]:
        """
        递归字符分割：按 \\n\\n → \\n → '. ' → ' ' 的优先级切分文本
        """
        if separators is None:
            separators = self.SEPARATORS

        if not text or len(text) <= chunk_size:
            return [text] if text.strip() else []

        # 找到当前层级的分隔符
        sep = separators[0]
        remaining_seps = separators[1:] if len(separators) > 1 else []

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # 如果单个 part 仍然超长，用更细的分隔符继续切
                if len(part) > chunk_size and remaining_seps:
                    sub_chunks = self._recursive_split(part, chunk_size, remaining_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current)

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        滑动窗口 Overlap：相邻 chunk 之间保留 overlap 字符的重叠区域，
        防止信息断层，确保上下文连贯。
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: List[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # 取前一个 chunk 末尾的 overlap 字符作为上下文前缀
                prev = chunks[i - 1]
                overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
                overlapped.append(overlap_text + " " + chunk)

        return overlapped

    # ------------------------------------------------------------------ #
    #  PDF 解析 Worker（Map 阶段）
    # ------------------------------------------------------------------ #

    def _extract_page_text(self, page_info: tuple) -> Dict[str, Any]:
        """单个 Worker：提取并清洗一页 PDF 文本（Map 阶段）"""
        pdf_path, page_num, total_pages = page_info
        worker_id = page_num % self.num_workers

        logger.info(f"[Worker-{worker_id}] Extracting page {page_num + 1}/{total_pages} from '{os.path.basename(pdf_path)}'")

        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            
            # 1. 提取基础纯文本
            raw_text = page.extract_text() or ""
            
            # 2. 高级版面分析：提取表格并转换为 Markdown 格式注入上下文
            tables = page.extract_tables()
            md_tables_text = ""
            if tables:
                for table in tables:
                    if not table: continue
                    md_table = "\n[PDF提取表格]:\n"
                    for i, row in enumerate(table):
                        # 清洗单元格内容中的换行符，防止 Markdown 结构断裂
                        row_clean = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
                        md_table += "| " + " | ".join(row_clean) + " |\n"
                        if i == 0:  # 插入表头隔离线
                            md_table += "|-" + "-|-".join(["-"] * len(row_clean)) + "-|\n"
                    md_tables_text += md_table + "\n"

        # 文本清洗：去除多余空白、控制字符
        cleaned = re.sub(r'\s+', ' ', raw_text).strip()
        # 去除常见的页眉页脚噪声（纯数字行、过短行）
        lines = raw_text.split('\n')
        cleaned_lines = [
            line.strip() for line in lines
            if line.strip() and len(line.strip()) > 5 and not line.strip().isdigit()
        ]
        cleaned = '\n'.join(cleaned_lines)
        
        # 将高价值结构化表格数据强行追加到当前页文本末尾，防止数据在分块时丢失
        if md_tables_text:
            cleaned += f"\n\n{md_tables_text}"

        return {
            "page_num": page_num,
            "text": cleaned,
            "source": os.path.basename(pdf_path),
            "worker_id": worker_id,
        }

    # ------------------------------------------------------------------ #
    #  主入口：解析单个 PDF
    # ------------------------------------------------------------------ #

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        解析单个 PDF 文件：
        1. 多线程并发提取各页文本（Map）
        2. 合并全文 → 递归分割 + Overlap（Reduce）
        3. 返回标准 chunk 列表
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []

        filename = os.path.basename(pdf_path)
        logger.info(f"========== Parsing PDF: '{filename}' ==========")

        # 获取页数
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        logger.info(f"Total pages: {total_pages}")

        # ---- Map 阶段：多线程并发提取各页文本 ----
        logger.info(f"[Map Phase] Distributing {total_pages} pages to {self.num_workers} workers...")
        page_tasks = [(pdf_path, i, total_pages) for i in range(total_pages)]
        page_results = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._extract_page_text, task): task[1] for task in page_tasks}
            for future in as_completed(futures):
                result = future.result()
                page_results.append(result)

        # 按页码排序后合并
        page_results.sort(key=lambda x: x["page_num"])
        full_text = "\n\n".join(r["text"] for r in page_results if r["text"])
        logger.info(f"[Map Phase] All pages extracted. Total characters: {len(full_text)}")

        # ---- Reduce 阶段：Small2Big 两级分块逻辑 ----
        logger.info(f"[Reduce Phase] Implementing Small2Big: Parent({self.parent_chunk_size}) -> Child({self.child_chunk_size})")
        
        # 1. 第一级：切出大的父分块（Parent Chunks）
        parent_texts = self._recursive_split(full_text, chunk_size=self.parent_chunk_size)
        
        final_chunks = []
        for p_idx, p_text in enumerate(parent_texts):
            # 2. 第二级：从每个父分块中，切出精细的子分块（Child Chunks）
            child_raw_texts = self._recursive_split(p_text, chunk_size=self.child_chunk_size)
            # 3. 对子分块应用滑动窗口，增强子块间的连续性
            child_overlapped_texts = self._apply_overlap(child_raw_texts)
            
            for c_idx, c_text in enumerate(child_overlapped_texts):
                chunk_id = f"{filename.replace('.pdf', '')}_P{p_idx:03d}_C{c_idx:03d}"
                final_chunks.append({
                    "doc_id": filename.replace('.pdf', ''),
                    "content": c_text, # 用于 Embedding 的子块内容
                    "text": c_text,
                    "parent_content": p_text, # 核心：检索到子块后，实际返回给 LLM 的完整父块内容
                    "metadata": {
                        "type": "child_node",
                        "source": filename,
                        "parent_id": f"P{p_idx:03d}",
                        "year": self._infer_year(filename, full_text),
                    },
                    "chunk_id": chunk_id,
                })

        logger.info(f"[Reduce Phase] Generated {len(final_chunks)} Small2Big nodes from {len(parent_texts)} parents.")

        return final_chunks

    # ------------------------------------------------------------------ #
    #  批量解析：整个目录下所有 PDF
    # ------------------------------------------------------------------ #

    def parse_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """批量解析目录下所有 PDF 文件"""
        if not os.path.isdir(dir_path):
            logger.error(f"Directory not found: {dir_path}")
            return []

        pdf_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')])
        logger.info(f"Found {len(pdf_files)} PDF files in '{dir_path}'")

        all_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(dir_path, pdf_file)
            chunks = self.parse_pdf(pdf_path)
            all_chunks.extend(chunks)

        logger.info(f"📚 Total: {len(all_chunks)} chunks from {len(pdf_files)} PDFs.")
        return all_chunks

    @staticmethod
    def _infer_year(filename: str, text: str) -> int:
        """从文件名或正文推断论文发表年份"""
        # 常见论文年份映射
        year_map = {
            "raft": 2014,
            "mapreduce": 2004,
            "gfs": 2003,
            "attention": 2017,
            "bert": 2018,
        }
        fn_lower = filename.lower()
        for key, year in year_map.items():
            if key in fn_lower:
                return year
        # 尝试从文本中提取年份
        import re
        years = re.findall(r'20[0-2]\d|199\d', text[:2000])
        if years:
            return int(max(years))
        return 2024


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = DeepDocumentParser()

    # 测试单个 PDF
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if os.path.isdir(data_dir):
        chunks = parser.parse_directory(data_dir)
        for c in chunks[:3]:
            print(f"\n--- {c['chunk_id']} (year={c['metadata']['year']}) ---")
            print(c['content'][:200])
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please run demo.py first to download sample PDFs.")
