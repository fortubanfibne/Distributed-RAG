# Distributed RAG - Development Log

## v0.3 (2026-02-20)
- DHT cache with LRU eviction + TTL lazy deletion
- Hybrid search: metadata filter + dense vector + Small2Big
- MD5 composite cache key to prevent cross-filter collisions
