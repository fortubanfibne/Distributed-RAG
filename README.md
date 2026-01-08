# Distributed-RAG

A production-grade distributed RAG (Retrieval-Augmented Generation) system built from scratch — designed to solve real-world enterprise knowledge base bottlenecks: **single-node memory overflow** and **high-frequency query throughput limitations**.

## Why I Built This

While scaling my previous AI SaaS products, I hit a hard wall: single-node vector retrieval couldn't handle growing data volumes, and LLM-as-a-Judge evaluation results were inconsistent. Instead of patching with off-the-shelf tools, I decided to go deep and build the infrastructure layer myself.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Query Pipeline                   │
│   User Query → DHT Cache → Hybrid Retrieval → LLM Gen  │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │   Consistent Hash Router   │  ← storage.py
         └──────┬──────────┬──────────┘
                │          │
         ┌──────▼──┐  ┌────▼─────┐
         │ Shard 0 │  │ Shard 1  │   ← LanceDB instances
         │ Raft ×3 │  │ Raft ×3  │   ← raft.py
         └─────────┘  └──────────┘
```

## Core Modules

### 1. Consistent Hash Sharding (`core/storage.py`)
- `ConsistentHashRing` maps `doc_id` (MD5) to shards via `bisect.bisect_right` — O(log n) routing
- 100 virtual nodes per shard to prevent data skew
- `add_shard()` triggers only ~1/n data migration, no global cache invalidation

### 2. Raft Consensus (`core/raft.py`)
- Full Raft state machine: Leader election, WAL pre-write logging, log replication
- Randomized heartbeat timeout (150–300ms) to prevent split-vote livelock
- Up-to-date log check on vote grants — guarantees no stale leader wins

### 3. DHT Cache Layer (`core/dht_cache.py`)
- Composite key: `MD5(query + sorted_filters)` — prevents cross-year cache collisions
- LRU eviction via `collections.OrderedDict` with O(1) `move_to_end()`
- TTL lazy deletion on read to avoid serving stale results

### 4. Hybrid Search (`core/retriever.py`)
- **Metadata pre-filter** → narrows search space before ANN
- **Dense retrieval**: BAAI/BGE-M3 1024-dim semantic vectors via LanceDB HNSW
- **Small2Big cascade**: retrieve fine-grained chunks, return full parent document to LLM

### 5. Distributed Inference Layer (`core/inference.py`)
- Adapter pattern: current implementation routes to SiliconFlow API (Qwen2.5-7B-Instruct)
- Pre-wired for Tensor Parallelism / Pipeline Parallelism / PagedAttention
- Swap `GeneratorAdapter` to plug into a real vLLM cluster with zero upstream changes

### 6. LLM-as-a-Judge Evaluation (`core/evaluator.py`)
- RAGAS-style 3-metric evaluation: **Faithfulness**, **Answer Relevancy**, **Context Precision**
- Judge model: Qwen2.5-7B at `temperature=0.0` for deterministic scoring
- Baseline comparison vs. Jaccard — demonstrates semantic vs. lexical evaluation gap

## Real vs. Simulated

| Component | Status | Notes |
|-----------|--------|-------|
| PDF Parsing + ETL | ✅ Real | `pdfplumber` + table-to-Markdown |
| Raft State Machine | ✅ Real | WAL, log replication, leader election |
| LanceDB Vector Store | ✅ Real | Disk-based, Apache Arrow, mmap |
| BGE-M3 Embedding | ✅ Real | Via SiliconFlow API |
| Qwen2.5-7B Generation | ✅ Real | Via SiliconFlow API |
| RAGAS Evaluation | ✅ Real | LLM-as-a-Judge with 3 metrics |
| RPC Transport | 🔵 Simulated | In-process method calls (interface-isolated) |
| Distributed Inference | 🔵 Simulated | Trace logs; plug-and-play with real vLLM |

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/fortubanfibne/Distributed-RAG.git
cd Distributed-RAG
pip install -r requirements.txt  # or: uv sync

# Set your API key
export SILICONFLOW_API_KEY=your_key_here

# Run the full pipeline demo
python demo.py
```

You'll see live trace logs showing Raft consensus, shard routing, hybrid retrieval, cache hits, and end-to-end evaluation scores.

## Evaluation Results

| Test Case | Faithfulness | Answer Relevancy | Context Precision |
|-----------|-------------|------------------|-------------------|
| Normal QA | 0.85 | 0.82 | 0.81 |
| Hallucination (injected) | 0.39 | 0.41 | 0.38 |

The system correctly identifies and low-scores hallucinated answers at the semantic level — something lexical metrics like Jaccard completely miss.

## Tech Stack

`Python` · `LanceDB` · `BAAI/BGE-M3` · `Qwen2.5-7B` · `SiliconFlow API` · `Apache Arrow` · `HNSW`

---

*Built as part of my independent AI infrastructure research — exploring what it actually takes to scale RAG beyond a single node.*
