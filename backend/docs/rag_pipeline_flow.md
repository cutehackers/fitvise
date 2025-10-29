# RAG Pipeline Visual Flow

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAG PIPELINE FLOW                                     │
│                     Documents → Searchable Intelligence                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Document Ingestion & Processing

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Documents │    │  Text Extraction │    │  Metadata      │
│                 │    │                 │    │  Enrichment     │
│ • PDF files     │───▶│ • Content       │───▶│ • Doc type      │
│ • Text files    │    │   parsing       │    │ • Creation date │
│ • Web content   │    │ • Structure     │    │ • Source        │
│ • Guides        │    │   preservation  │    │ • Department    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │   Intelligent Chunking  │
                    │                         │
                    │ • 200-500 token chunks  │
                    │ • Context boundaries    │
                    │ • Semantic coherence    │
                    │ • Unique chunk IDs      │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Processed Chunks      │
                    │   + Metadata            │
                    └─────────────────────────┘
```

## Phase 2: Embedding Generation & Storage

```
                    ┌─────────────────────────┐
                    │   Processed Chunks      │
                    │   + Metadata            │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Embedding Generation   │
                    │                         │
                    │ • Sentence Transformers │
                    │ • 384-dim vectors       │
                    │ • Semantic preservation │
                    │ • Performance cache     │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Embeddings + Chunks   │
                    │   Ready for Storage     │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │     Weaviate Store      │
                    │                         │
                    │ • Vector database       │
                    │ • DocumentChunk schema  │
                    │ • Metadata indexing     │
                    │ • Similarity search     │
                    └─────────────────────────┘
```

## Phase 3: Semantic Search & Retrieval

```
                    ┌─────────────────────────┐
                    │      User Query         │
                    │   "exercises for        │
                    │   lower back pain"      │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Query Processing      │
                    │                         │
                    │ • Natural language      │
                    │ • Embed query           │
                    │ • Apply filters         │
                    │ • Cache results         │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Vector Similarity     │
                    │   Search                │
                    │                         │
                    │ • Cosine similarity     │
                    │ • Ranked results        │
                    │ • Relevance scoring     │
                    │ • Top-K selection       │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Search Results        │
                    │                         │
                    │ • Relevant chunks       │
                    │ • Similarity scores     │
                    │ • Document metadata     │
                    │ • Quality labels        │
                    └─────────────────────────┘
```

## Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                INPUT SOURCES                                    │
│                                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    PDFs     │  │   Text      │  │    Web      │  │   Guides   │              │
│  │  Documents  │  │   Files     │  │  Content    │  │  Manuals   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────┬───────────────────┬───────────────────┬───────────────────────┘
                  │                   │                   │
                  ▼                   ▼                   ▼
              ┌─────────────────────────────────────────────────────┐
              │            DOCUMENT INGESTION ENGINE                │
              │                                                     │
              │  • Text extraction • Structure preservation         │
              │  • Metadata enrichment • Quality validation         │
              └─────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
              ┌─────────────────────────────────────────────────────┐
              │            INTELLIGENT CHUNKING                     │
              │                                                     │
              │  • 200-500 token chunks • Context boundaries        │
              │  • Semantic coherence • Unique IDs                  │
              └─────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
              ┌─────────────────────────────────────────────────────┐
              │          EMBEDDING GENERATION                       │
              │                                                     │
              │  • Sentence Transformers • 384-dim vectors          │
              │  • Performance cache • Semantic preservation        │
              └─────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
              ┌─────────────────────────────────────────────────────┐
              │            WEAVIATE VECTOR STORE                    │
              │                                                     │
              │  • DocumentChunk schema • Metadata indexing         │
              │  • Similarity search • Scale to millions            │
              └─────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
              ┌─────────────────────────────────────────────────────┐
              │            SEMANTIC SEARCH API                      │
              │                                                     │
              │  POST /api/v1/rag/search/semantic                   │
              │  • Natural language queries • Vector similarity     │
              │  • Filtering • Ranked results • Performance         │
              └─────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
              ┌─────────────────────────────────────────────────────┐
              │              SEARCH RESULTS                         │
              │                                                     │
              │  • Relevant chunks • Similarity scores              │
              │  • Document metadata • Quality labels               │
              │  • Processing metrics • Source citations            │
              └─────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INTEGRATION POINTS                                    │
│                                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   LLM       │  │   Chat      │  │ Analytics   │  | Content    │              │
│  │ Services    │  │ Interfaces  │  │   Systems   │  | Recommender│              │
│  │             │  │             │  │             │  |   Engines  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Specifications

### Processing Pipeline
```
Raw Documents → Text Extraction → Metadata Enrichment → Intelligent Chunking
     ↓
Embedding Generation → Weaviate Storage → Semantic Search API → Search Results
```

### Performance Metrics
```
┌─────────────────┬─────────────────┬───────────────────┐
│    Phase        │   Speed         │   Quality         │
├─────────────────┼─────────────────┼───────────────────┤
│   Ingestion     │ 10K chunks/hr   │ 200-500 tokens    │
│   Embedding     │ <500ms/chunk    │ 384-dim vectors   │
│   Search        │ <200ms/query    │ Cosine similarity │
│   Total         │ <500ms/response │ Ranked results    │
└─────────────────┴─────────────────┴───────────────────┘
```

### API Flow
```
User Query → Query Embedding → Vector Search → Result Ranking → Response
     ↓              ↓              ↓              ↓            ↓
Natural Language → 384-dim vector → Similarity match → Score sort → JSON
```

## Key Integration Points

### Input Sources
- **Document Repositories**: PDF libraries, text files, web scrapers
- **Content Management**: Fitness guides, workout plans, nutrition data
- **External APIs**: Health databases, exercise libraries

### Output Consumers
- **LLM Services**: Context for answer generation
- **Chat Interfaces**: Real-time user assistance
- **Analytics**: Usage patterns and search optimization
- **Recommendation**: Related content discovery

This visual flow represents the complete RAG pipeline from raw document ingestion through semantic search, enabling intelligent content retrieval for fitness-related queries.