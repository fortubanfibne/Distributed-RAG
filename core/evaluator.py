import logging
import time
import os
import httpx
import json

# ========== 硅基流动 API 配置 ==========
SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-lthsvztszxotjlcxocgfgitztcpxtcnhkwklzuejbbfstozd"
)
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
EVAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def _llm_eval(prompt: str) -> float:
    """调用大模型进行打分，返回 0.0 ~ 1.0 之间的分数"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EVAL_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严格的评分裁判。请根据用户的要求进行评分，只能输出一个数字，比如 0.8，不要输出任何额外解释。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{SILICONFLOW_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            result_text = resp.json()["choices"][0]["message"]["content"].strip()
            
            # 尝试从回复中提取数字
            try:
                score = float(''.join(c for c in result_text if c.isdigit() or c == '.'))
                # 确保分数在 0-1 之间（如果模型输出了 0-10 或 0-100，将其归一化）
                if score > 10.0:
                    score = score / 100.0
                elif score > 1.0:
                    score = score / 10.0
                return float(round(min(max(score, 0.0), 1.0), 3))
            except ValueError:
                return 0.5  # 解析失败给个基础分
    except Exception as e:
        logger.error(f"[Evaluator] LLM Eval error: {e}")
        return 0.0

logger = logging.getLogger("RAG-Evaluator")


class RAGEvaluator:
    """
    参考 DREAM (Distributed RAG Experimentation Framework) 的评估体系，
    模拟 Ragas 框架对 RAG 系统的三项核心量化指标。

    面试话术：
        "老师，一个 RAG 系统的好坏不能只靠人眼看日志，需要可量化的指标来驱动迭代。
        我参考了业界主流的 Ragas 评测框架，在系统中集成了三项核心质量指标：

        ① Faithfulness（忠实度）：衡量答案是否完全来自于我检索到的上下文，
          而不是模型自己'幻觉'出来的。用来检测 LLM 的幻觉率。

        ② Answer Relevancy（答案相关性）：衡量生成的答案是否真正回答了用户的问题。
          一个答案可能忠实于上下文但完全答非所问。

        ③ Context Precision（上下文精准度）：衡量检索回来的文档有多少是真正有用的，
          避免'检索到一堆废话'导致 LLM 被噪音干扰。

        通过这三项指标，我可以定量对比不同检索策略（如向量 vs 混合检索）
        或不同分块粒度的优劣，这正是 DREAM 等工业级框架的核心设计思想。"
    """

    def __init__(self):
        logger.info("[Evaluator] Initializing RAG Quality Evaluator (Ragas-style)...")

    def _calc_faithfulness(self, answer: str, context: str) -> float:
        """
        忠实度 (Faithfulness)：答案是否完全来自于检索到的上下文（无幻觉）。
        经过脱敏的上下文与答案可能在字面上不完全一致，但语义应当一致。
        """
        prompt = (
            f"请评估以下【回答】是否忠实地基于以下【上下文】得出。即便存在人名词汇等脱敏替换，只要语义能对应即可。\n"
            f"上下文：{context}\n回答：{answer}\n"
            f"请给出一个 0.0 到 1.0 之间的分数（1.0表示完全基于上下文，0.0表示完全是幻觉）。"
        )
        return _llm_eval(prompt)


    def _calc_answer_relevancy(self, question: str, answer: str) -> float:
        """
        答案相关性 (Answer Relevancy)：评估答案是否直接回答了用户的问题。
        """
        prompt = (
            f"请评估以下【回答】是否精准、直接地回答了【问题】。\n"
            f"问题：{question}\n回答：{answer}\n"
            f"请给出一个 0.0 到 1.0 之间的分数（1.0表示完美回答，0.0表示答非所问）。"
        )
        return _llm_eval(prompt)


    def _calc_context_precision(self, answer: str, context: str) -> float:
        """
        上下文精准度 (Context Precision)：检索出的上下文有多大比例对生成答案是有用的。
        """
        prompt = (
            f"请评估以下【片段/上下文】中有多少比例的信息对推导出最终【回答】是真正有用的（信息密度及精准度）。\n"
            f"片段/上下文：{context}\n最终回答：{answer}\n"
            f"请给出一个 0.0 到 1.0 之间的分数（1.0表示给的上下文全是干货且都被用到了，0.0表示全是废话无用信息）。"
        )
        return _llm_eval(prompt)


    def evaluate(self, question: str, answer: str, context: str) -> dict:
        """
        对一次 RAG 问答进行三维度评估，输出 Ragas 风格的评测报告。
        """
        logger.info("=" * 55)
        logger.info("[Evaluator] ===== RAG Quality Evaluation Report =====")
        logger.info(f"[Evaluator] Question: '{question}'")
        time.sleep(0.3)

        faithfulness      = self._calc_faithfulness(answer, context)
        answer_relevancy  = self._calc_answer_relevancy(question, answer)
        context_precision = self._calc_context_precision(answer, context)

        # 综合评分（加权均值）
        composite = round(
            faithfulness * 0.4 + answer_relevancy * 0.35 + context_precision * 0.25, 3
        )

        logger.info(f"[Evaluator] ✅ Faithfulness      : {faithfulness:.3f}  "
                    f"(答案是否忠实于上下文，无幻觉)")
        logger.info(f"[Evaluator] ✅ Answer Relevancy  : {answer_relevancy:.3f}  "
                    f"(答案是否真正回答了问题)")
        logger.info(f"[Evaluator] ✅ Context Precision : {context_precision:.3f}  "
                    f"(检索文档的有效利用率)")
        logger.info(f"[Evaluator] 🏆 Composite Score   : {composite:.3f}")
        logger.info("[Evaluator] ================================================")

        return {
            "faithfulness":       faithfulness,
            "answer_relevancy":   answer_relevancy,
            "context_precision":  context_precision,
            "composite_score":    composite,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    evaluator = RAGEvaluator()
    report = evaluator.evaluate(
        question="孟老师的研究方向是什么？",
        answer="基于上下文，相关教授的研究方向涵盖数据库系统和Web数据管理。",
        context="数据库系统的底座是存储，孟老师专注于Web数据管理领域。",
    )
    print("\nEvaluation Report:", report)
