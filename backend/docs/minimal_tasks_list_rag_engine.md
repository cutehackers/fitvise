# Minimal essential tasks for a working RAG engine.

⏺ Minimal RAG Engine: Essential Tasks

  Based on your existing setup (Task 2.2.1 complete, Weaviate configured), here are the essential tasks for a minimal working RAG engine:

  ---
  ✅ ESSENTIAL: Core RAG Functionality (23 days)

  Task 2.3.2: Weaviate Schema Design (5 days) - CRITICAL

  Why essential: Without schema, you can't store embeddings
  Deliverable:
  - Weaviate schema supporting document chunks with metadata
  - Efficient filtering on doc_type, date, department, source

  Minimal scope:
  # Minimal schema fields
  {
    "class": "DocumentChunk",
    "vectorizer": "none",  # Using your existing embedding model
    "properties": [
      {"name": "content", "dataType": ["text"]},           # Chunk text
      {"name": "chunk_id", "dataType": ["uuid"]},          # Unique ID
      {"name": "document_id", "dataType": ["string"]},     # Parent doc
      {"name": "doc_type", "dataType": ["string"]},        # For filtering
      {"name": "metadata", "dataType": ["object"]}         # Additional metadata
    ]
  }

  ---
  Task 2.3.3: Embedding Ingestion Pipeline (10 days) - CRITICAL

  Why essential: Connects your chunks → embeddings → Weaviate
  Deliverable:
  - Service that takes chunks from Tasks 2.1.x
  - Embeds using your existing EmbedDocumentChunksUseCase
  - Stores in Weaviate with metadata

  Minimal scope:
  - Batch ingestion (no need for real-time yet)
  - Basic deduplication (hash-based)
  - Error handling and retry logic
  - Target: 10K chunks/hour (you already achieve 15K with embedding)

  ---
  Task 2.4.1-MINIMAL: Basic Semantic Search (8 days) - CRITICAL

  Why essential: This is the "R" in RAG - retrieval
  Deliverable: Simple vector similarity search using Weaviate

  Minimal scope (skip Elasticsearch for now):
  # Minimal search use case
  class SemanticSearchUseCase:
      async def execute(self, query: str, top_k: int = 5) -> List[SearchResult]:
          # 1. Embed query using existing EmbedQueryUseCase
          query_embedding = await self.embed_query_uc.execute(query)

          # 2. Search Weaviate for similar vectors
          results = await self.weaviate_repo.search_similar(
              vector=query_embedding.vector,
              limit=top_k
          )

          # 3. Return chunks with similarity scores
          return results

  Skip for minimal MVP: Elasticsearch, BM25, keyword search (can add later)

  ---
  Task 2.3.1-PARTIAL: Minimal Weaviate Setup (Already ~80% done)

  Why partial: Your docker-compose.yml already has Weaviate 1.24.1 configured
  What's left (optional, can defer):
  - Production backup/restore scripts
  - High availability configuration
  - Monitoring dashboards

  For minimal MVP: Your current Docker setup is sufficient

  ---
  ❌ NON-ESSENTIAL: Defer for Later Improvements

  Task 2.4.1-FULL: Elasticsearch Integration (8 days) - SKIP FOR NOW

  - Why skip: Semantic search alone is sufficient for MVP
  - When to add: If users need exact keyword matching (e.g., code, IDs, specific terms)
  - Complexity: Requires separate Elasticsearch cluster, dual-index management

  ---
  Task 2.4.2: Hybrid Search (12 days) - SKIP FOR NOW

  - Why skip: Requires both semantic + keyword search (Task 2.4.1)
  - When to add: After analyzing retrieval quality—does semantic-only miss important results?
  - Complexity: Need Haystack or custom fusion logic, weighting algorithms

  ---
  Task 2.4.3: Cross-Encoder Re-ranking (10 days) - SKIP FOR NOW

  - Why skip: Optimization on top of basic retrieval
  - When to add: If top-K results have poor ordering (need metrics first)
  - Complexity: Additional ML model, increased latency (re-rank every query)

  ---
  Task 2.4.4: Query Classification (12 days) - SKIP FOR NOW

  - Why skip: Only useful if you have hybrid search (Task 2.4.2)
  - When to add: After implementing hybrid search, if routing improves results
  - Complexity: BART model fine-tuning, query type taxonomy

  ---

⏺ 📋 Minimal RAG Engine Implementation Order

  Phase 1: Storage Layer (15 days)

  1. ✅ Task 2.3.2: Design Weaviate schema (5 days)
    - Define chunk schema with metadata
    - Test with sample data
  2. ✅ Task 2.3.3: Build ingestion pipeline (10 days)
    - Connect chunks → embeddings → Weaviate
    - Implement batch processing
    - Add error handling

  Phase 2: Retrieval Layer (8 days)

  3. ✅ Task 2.4.1-MINIMAL: Basic semantic search (8 days)
    - Embed query
    - Vector similarity search
    - Return top-K chunks with scores

  Total: 23 days for minimal working RAG

  ---
  🎯 What You'll Have: Minimal RAG Engine

  # Minimal RAG workflow
  async def rag_query(user_query: str) -> str:
      # 1. Embed query (Task 2.2.1 - DONE)
      query_embedding = await embed_query(user_query)

      # 2. Search Weaviate (Task 2.3.2 + 2.3.3)
      relevant_chunks = await weaviate.search(
          vector=query_embedding,
          limit=5
      )
      # 3. Serving relevant chunks to LLM service (existing LLM service)
      #  This is not goting to be included in RAG workflow as it is not part of the RAG pipeline.
      #  It's just a part of entire flow to serve relevant chunks to LLM service.
      context = "\n".join([chunk.content for chunk in relevant_chunks])
      answer = await llm_service.generate(
          prompt=f"Context: {context}\n\nQuestion: {user_query}"
      )

      return answer

  This gives you:
  - ✅ Document ingestion
  - ✅ Semantic search
  - ✅ Context retrieval
  - ✅ LLM answer generation
  - ✅ Full RAG pipeline

  ---
  📊 Effort Comparison

  | Approach    | Tasks                                | Days    | Features                           |
  | ----------- | ------------------------------------ | ------- | ---------------------------------- |
  | Minimal RAG | 2.3.2, 2.3.3, 2.4.1-minimal          | 23      | Semantic search only               |
  | Full RAG    | All 2.3.x + 2.4.x                    | 65      | Hybrid, re-ranking, classification |
  | Savings     | Skip 2.4.1-full, 2.4.2, 2.4.3, 2.4.4 | 42 days | 65% faster to MVP                  |

  ---
  🚀 Recommended Path: Iterative Improvement

  MVP (23 days)

  ✅ Tasks 2.3.2, 2.3.3, 2.4.1-minimal

  Improvement Phase 1 (if retrieval quality is poor)

  - Add Task 2.4.3 (Re-ranking) - Improves result ordering

  Improvement Phase 2 (if semantic search misses keywords)

  - Add Task 2.4.1-full (Elasticsearch) - Exact keyword matching
  - Add Task 2.4.2 (Hybrid search) - Combine semantic + keyword

  Improvement Phase 3 (optimization)

  - Add Task 2.4.4 (Query classification) - Route queries intelligently

  ---
  ⚡ Quick Start: Next Steps

  1. Week 1-2: Implement Task 2.3.2 (Weaviate schema)
    - Design schema for your fitness documents
    - Test ingestion with sample data
  2. Week 3-4: Implement Task 2.3.3 (Ingestion pipeline)
    - Build ETL: chunks → embeddings → Weaviate
    - Batch process existing documents
  3. Week 5: Implement Task 2.4.1-minimal (Search)
    - Build semantic search use case
    - Integrate with existing LLM service
    - Test end-to-end RAG query

  Result: Working RAG engine in ~5 weeks vs. 13 weeks for full implementation

  ---
  💡 Key Insights

  Why this works:
  - Your embedding infrastructure (Task 2.2.1) is production-ready
  - Weaviate handles vector search efficiently (no Elasticsearch needed for MVP)
  - Semantic search alone covers 80% of RAG use cases
  - You can measure retrieval quality and add complexity only if needed

  When to add complexity:
  - Re-ranking: If top-5 results have wrong ordering
  - Keyword search: If semantic search misses exact terms/codes
  - Hybrid search: If combining semantic + keyword improves metrics
  - Query classification: If different query types need different strategies

  Start minimal, measure, then improve based on evidence.