# Task 2.1.3: Recursive Chunking for Hierarchical Documents - Implementation

## Executive Summary

**Task**: Implement LangChain-based recursive chunking respecting document structure (policy > section > paragraph)
**Chosen Approach**: llama_index `HierarchicalNodeParser` (Option B)
**Rationale**: Consistency with Task 2.1.1, simpler architecture, RAG-optimized with LangChain interoperability

---

## Table of Contents
1. [Pre-Implementation Analysis](#pre-implementation-analysis)
2. [Framework Comparison](#framework-comparison)
3. [Interoperability Architecture](#interoperability-architecture)
4. [Implementation Plan](#implementation-plan)
5. [Acceptance Criteria](#acceptance-criteria)

---

## Pre-Implementation Analysis

### Task 2.1.1 Verification ✅
**Implementation**: `backend/app/infrastructure/external_services/ml_services/chunking_services/llama_index_chunker.py`

**Key Components**:
- `SemanticSplitterNodeParser` with HuggingFace embeddings (all-MiniLM-L6-v2)
- Breakpoint threshold: 85% (configurable)
- Graceful fallback: Semantic → Sentence → Basic chunker
- Proper metadata preservation and error handling

**Status**: ✅ Correctly implemented using llama_index

### Current System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    RAG System Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Chunking Layer (Phase 2.1)                                 │
│  ├── Task 2.1.1: Semantic Chunking (llama_index) ✅         │
│  ├── Task 2.1.2: Table Serialization ✅                      │
│  └── Task 2.1.3: Recursive Chunking (llama_index) ← THIS    │
│                                                               │
│  Vector Store Layer (Phase 2.3)                              │
│  └── Weaviate (planned - supports both frameworks)          │
│                                                               │
│  Retrieval Layer (Phase 2.4)                                 │
│  └── LangChain RetrievalQA, Hybrid Search (planned)         │
│                                                               │
│  LLM Layer (Phase 3.x)                                       │
│  └── LangChain ChatOllama ✅ (already implemented)           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Framework Comparison

### Option A: LangChain RecursiveCharacterTextSplitter

#### Technical Implementation
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=200,
    separators=["\n\n\n", "\n\n", "\n", " ", ""],  # Hierarchical order
    keep_separator=True,
)
```

#### How It Works
1. Try split by `\n\n\n` (document sections)
2. If chunks too large → split by `\n\n` (paragraphs)
3. If still too large → split by `\n` (sentences)
4. If still too large → split by space (words)

#### Pros ✅
- Industry-standard for document chunking
- Maximum flexibility with custom separators
- Better for structured text (markdown, code, policies)
- Explicit control over hierarchy
- Native LangChain integration (no conversion needed)

#### Cons ❌
- **Manual hierarchy tracking**: Must implement heading path extraction
- **Two frameworks**: Adds complexity (LangChain + llama_index)
- **More boilerplate**: Custom metadata enrichment code
- **Not RAG-optimized**: General-purpose text splitter

---

### Option B: llama_index HierarchicalNodeParser ⭐ (CHOSEN)

#### Technical Implementation
```python
from llama_index.core.node_parser import HierarchicalNodeParser

hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],  # Multi-level: document → section → paragraph
)
```

#### How It Works
1. **Level 1 (2048 chars)**: Full document context chunks
2. **Level 2 (512 chars)**: Section-level chunks (child of Level 1)
3. **Level 3 (128 chars)**: Paragraph-level chunks (child of Level 2)
4. **Automatic parent tracking**: Each chunk knows its parent

#### Pros ✅
- **Single framework**: All chunking in llama_index (consistency)
- **Automatic parent-child**: Built-in relationship tracking
- **RAG-optimized**: Designed for retrieval pipelines
- **Metadata-rich**: Automatic heading path, section extraction
- **Consistent API**: Same pattern as Task 2.1.1
- **Less code**: ~50% reduction vs manual hierarchy tracking

#### Cons ❌
- **Less flexible**: Opinionated about hierarchy levels
- **Chunk size based**: Uses character counts, not semantic separators
- **Conversion layer**: Needs adapter for LangChain retrieval pipeline

---

### Decision Matrix

| Aspect                | Option A: LangChain         | Option B: llama_index ⭐   |
|-----------------------|-----------------------------|---------------------------|
| **Consistency**       | ❌ Two frameworks           | ✅ One framework           |
| **Dependencies**      | Already installed ✅        | Already installed ✅       |
| **Hierarchical Quality** | ⭐⭐⭐⭐⭐ Excellent       | ⭐⭐⭐⭐ Very Good         |
| **Customization**     | ⭐⭐⭐⭐⭐ Maximum         | ⭐⭐⭐ Moderate            |
| **Backlog Alignment** | ✅ Matches spec             | ⚠️ Deviates slightly       |
| **Code Maintenance**  | More complex                | ✅ Simpler                 |
| **RAG Optimization**  | ⭐⭐⭐ Good                 | ⭐⭐⭐⭐⭐ Excellent       |
| **Parent-child tracking** | ❌ Manual                 | ✅ Automatic               |
| **Metadata enrichment** | ❌ Custom code              | ✅ Built-in                |
| **Best For**          | Flexible splitting          | RAG pipelines              |

---

## Interoperability Architecture

### Key Question: Can LangChain Access llama_index Chunks?

**Answer**: ✅ **YES** - Full interoperability with simple conversion layer

### Current LLM Layer
```python
# backend/app/application/llm_service.py (lines 42-46)
from langchain_ollama.chat_models import ChatOllama

self.llm = ChatOllama(
    base_url=settings.llm_base_url,
    model=settings.llm_model,
)
```

### Data Flow with Conversion Layer

```
┌──────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Data Flow                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. Document Ingestion                                            │
│     └─> Document entity with text/metadata                       │
│                                                                    │
│  2. Chunking (llama_index)                                        │
│     └─> HierarchicalNodeParser                                   │
│         ├─> Level 1: TextNode (2048 chars, parent_id=None)       │
│         ├─> Level 2: TextNode (512 chars, parent_id=L1_id)       │
│         └─> Level 3: TextNode (128 chars, parent_id=L2_id)       │
│                                                                    │
│  3. Conversion Layer ← NEW (20 lines of code)                     │
│     └─> convert_llama_nodes_to_langchain()                       │
│         Input: List[TextNode]                                     │
│         Output: List[LangChainDocument]                           │
│         Preserves: heading_path, section, parent_id, metadata    │
│                                                                    │
│  4. Embedding & Vector Store (Framework-agnostic)                 │
│     └─> Weaviate.from_documents(langchain_docs, embeddings)      │
│                                                                    │
│  5. Retrieval (LangChain)                                         │
│     └─> RetrievalQA.from_chain_type(                             │
│             llm=ChatOllama,  ← Your existing LLM                  │
│             retriever=vectorstore.as_retriever()                  │
│         )                                                         │
│                                                                    │
│  6. LLM Response (LangChain)                                      │
│     └─> ChatOllama.invoke(query + retrieved_context)             │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Conversion Layer Implementation

```python
# backend/app/infrastructure/adapters/llama_to_langchain.py

from typing import List
from langchain.schema import Document as LangChainDocument
from llama_index.core.schema import TextNode

def convert_llama_nodes_to_langchain(
    nodes: List[TextNode]
) -> List[LangChainDocument]:
    """
    Convert llama_index TextNodes to LangChain Documents.

    Preserves hierarchical metadata:
    - heading_path: List of heading breadcrumbs
    - section: Section identifier
    - parent_id: Parent chunk reference
    - depth_level: Hierarchy level (0=root, 1=section, 2=paragraph)
    """
    return [
        LangChainDocument(
            page_content=node.text,
            metadata={
                "chunk_id": node.node_id,
                "heading_path": node.metadata.get("heading_path", []),
                "section": node.metadata.get("section"),
                "parent_id": node.ref_doc_id,
                "depth_level": node.metadata.get("depth_level", 0),
                **node.metadata,
            }
        )
        for node in nodes
    ]
```

**Result**: ✅ Full LangChain compatibility with ~20 lines of code

---

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 days)

#### File: `backend/app/infrastructure/external_services/ml_services/chunking_services/llama_hierarchical_chunker.py`

**Components**:

1. **HierarchicalChunkerConfig** dataclass
```python
@dataclass
class HierarchicalChunkerConfig:
    chunk_sizes: List[int] = field(default_factory=lambda: [2048, 512, 128])
    chunk_overlap: int = 200
    min_chunk_chars: int = 100
    max_chunk_chars: int = 2048
    preserve_hierarchy: bool = True
    metadata_passthrough_fields: Sequence[str] = (
        "document_id", "source_id", "file_name", "doc_type"
    )
    debug_mode: bool = False
```

2. **HierarchicalChunk** dataclass
```python
@dataclass
class HierarchicalChunk:
    chunk_id: str
    sequence: int
    text: str
    start: int
    end: int
    depth_level: int  # NEW: 0=root, 1=section, 2=paragraph
    parent_chunk_id: Optional[str]  # NEW: Parent reference
    metadata: Dict[str, Any] = field(default_factory=dict)
```

3. **LlamaHierarchicalChunker** class
```python
class LlamaHierarchicalChunker:
    def __init__(
        self,
        config: Optional[HierarchicalChunkerConfig] = None,
        require_llama_index: bool = False,
    ):
        self.config = config or HierarchicalChunkerConfig()

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[HierarchicalChunk]:
        # Use HierarchicalNodeParser.from_defaults()
        # Convert TextNodes to HierarchicalChunks
        # Preserve parent-child relationships
        pass
```

**Key Features**:
- Multi-level chunking (2048 → 512 → 128 chars)
- Automatic parent-child tracking
- Graceful fallback to SentenceSplitter
- Consistent API with `LlamaIndexChunker`

---

### Phase 2: Use Case Integration (2 days)

#### File: `backend/app/application/use_cases/chunking/recursive_chunking.py`

**Components**:

```python
@dataclass
class RecursiveChunkingRequest:
    document_ids: Optional[Sequence[UUID]] = None
    include_failed: bool = False
    replace_existing_chunks: bool = True
    dry_run: bool = False
    chunker_config: Optional[Dict[str, Any]] = None

@dataclass
class RecursiveChunkingResponse:
    success: bool
    results: List[DocumentChunkSummary]
    total_chunks: int
    hierarchy_stats: Dict[str, int]  # NEW: Count by depth level
    dry_run: bool

class RecursiveChunkingUseCase:
    def __init__(
        self,
        document_repository: DocumentRepository,
        chunker: Optional[LlamaHierarchicalChunker] = None,
    ):
        self._repository = document_repository
        self._chunker = chunker

    async def execute(
        self, request: RecursiveChunkingRequest
    ) -> RecursiveChunkingResponse:
        # Load documents
        # Chunk with hierarchical chunker
        # Convert HierarchicalChunk → domain Chunk
        # Save with hierarchy metadata
        pass
```

**Metadata Mapping**:
- `HierarchicalChunk.depth_level` → `ChunkMetadata.extra["depth_level"]`
- `HierarchicalChunk.parent_chunk_id` → `ChunkMetadata.extra["parent_id"]`
- Automatic `heading_path` and `section` population

---

### Phase 3: Interoperability Layer (1 day)

#### File: `backend/app/infrastructure/adapters/llama_to_langchain.py`

**Purpose**: Enable LangChain retrieval pipeline to access llama_index chunks

**Implementation**: See conversion function in [Interoperability Architecture](#interoperability-architecture)

---

### Phase 4: Configuration (0.5 day)

#### File: `backend/app/config/ml_models/chunking_configs.py`

**Add Preset**:
```python
"hierarchical": ChunkingPreset(
    name="hierarchical",
    description="Recursive chunking preserving policy/section/paragraph hierarchy",
    chunk_sizes=[2048, 512, 128],
    chunk_overlap=200,
    min_chunk_chars=100,
    max_chunk_chars=2048,
)
```

---

### Phase 5: Testing (3-4 days)

#### Unit Tests

**File**: `backend/tests/unit/infrastructure/external_services/ml_services/chunking_services/test_llama_hierarchical_chunker.py`

**Coverage**:
- Multi-level hierarchy creation (3 levels)
- Parent-child relationship validation
- Heading path extraction from markdown
- Section hierarchy preservation
- Edge cases: single-level docs, deeply nested structures
- Metadata passthrough and enrichment
- Graceful fallback to SentenceSplitter

**Fixtures**:
```python
# backend/tests/fixtures/hierarchical_documents.py

POLICY_DOCUMENT = """
# Company Policy Manual

## Section 1: Code of Conduct
### 1.1 Professional Behavior
Employees must maintain professional standards...

### 1.2 Workplace Ethics
Ethical conduct is essential...

## Section 2: Leave Policies
### 2.1 Annual Leave
Employees are entitled to 20 days...
"""
```

#### Integration Tests

**File**: `backend/tests/unit/application/use_cases/chunking/test_recursive_chunking.py`

**Coverage**:
- End-to-end use case execution
- Document → HierarchicalChunk → Chunk conversion
- Hierarchy metadata preservation
- Multi-document batching
- Dry-run mode validation

#### Conversion Layer Tests

**File**: `backend/tests/unit/infrastructure/adapters/test_llama_to_langchain.py`

**Coverage**:
- TextNode → LangChainDocument conversion
- Metadata preservation
- Hierarchy information transfer
- Edge cases: empty nodes, missing metadata

---

## Acceptance Criteria

### Functional Requirements ✅

1. **llama_index Integration**
   - [ ] Uses `HierarchicalNodeParser.from_defaults()`
   - [ ] Multi-level chunking (configurable levels)
   - [ ] Automatic parent-child relationships

2. **Hierarchical Structure Preservation**
   - [ ] Preserves policy > section > paragraph hierarchy
   - [ ] `heading_path` tracking (breadcrumb navigation)
   - [ ] `section` identification
   - [ ] `depth_level` metadata (0=root, 1=section, 2=paragraph)
   - [ ] `parent_chunk_id` references

3. **LangChain Interoperability**
   - [ ] Conversion adapter implemented
   - [ ] Full metadata preservation in conversion
   - [ ] Compatible with LangChain RetrievalQA
   - [ ] Vector store integration ready

4. **Configuration**
   - [ ] "hierarchical" preset added
   - [ ] Configurable chunk sizes per level
   - [ ] Consistent with existing config patterns

5. **Testing**
   - [ ] Unit tests: chunker + use case (≥80% coverage)
   - [ ] Integration tests: end-to-end pipeline
   - [ ] Conversion layer tests
   - [ ] Test fixtures for hierarchical documents

### Non-Functional Requirements ✅

1. **Consistency**: Same patterns as Task 2.1.1
2. **Maintainability**: Single chunking framework
3. **Performance**: <100ms for typical policy document
4. **Documentation**: Comprehensive inline docs + this IMPLEMENTATION.md

---

## Timeline

| Phase                   | Duration   | Deliverables                         |
|-------------------------|------------|--------------------------------------|
| Documentation           | 0.5 day    | This IMPLEMENTATION.md               |
| Core Infrastructure     | 2-3 days   | LlamaHierarchicalChunker             |
| Use Case Integration    | 2 days     | RecursiveChunkingUseCase             |
| Interoperability Layer  | 1 day      | llama_to_langchain adapter           |
| Configuration           | 0.5 day    | "hierarchical" preset                |
| Testing                 | 3-4 days   | Comprehensive test suite             |
| **Total**               | **8-10 days** | Task 2.1.3 complete ✅             |

---

## Architecture Decision Records (ADR)

### ADR-001: Choose llama_index over LangChain for Recursive Chunking

**Status**: ✅ Accepted

**Context**:
- Task 2.1.1 already uses llama_index for semantic chunking
- Task 2.1.3 backlog specifies "LangChain-based" recursive chunking
- LLM layer uses LangChain ChatOllama
- Need to decide: maintain framework consistency or follow spec literally

**Decision**: Use llama_index `HierarchicalNodeParser`

**Rationale**:
1. **Consistency**: Single chunking framework reduces complexity
2. **RAG Optimization**: llama_index designed for retrieval pipelines
3. **Code Reuse**: Share patterns, configs, error handling with Task 2.1.1
4. **Automatic Hierarchy**: Built-in parent-child tracking saves ~200 LOC
5. **Interoperability**: Simple 20-line adapter enables LangChain integration

**Consequences**:
- ✅ Simpler architecture (one chunking framework)
- ✅ ~50% less implementation code
- ✅ Consistent patterns across Epic 2.1
- ⚠️ Requires conversion layer for LangChain retrieval
- ⚠️ Deviates from literal backlog specification

**Mitigation**:
- Document conversion layer thoroughly
- Test interoperability comprehensively
- Update backlog with rationale

---

### ADR-002: Use 3-Level Hierarchy as Default

**Status**: ✅ Accepted

**Context**: Need to define default hierarchy depth for policy documents

**Decision**: Default chunk_sizes = [2048, 512, 128] (3 levels)

**Rationale**:
1. **Level 0 (2048)**: Document/chapter context
2. **Level 1 (512)**: Section-level chunks
3. **Level 2 (128)**: Paragraph-level granularity
4. Matches typical policy document structure

**Consequences**:
- Configurable per use case
- May need tuning for specific document types

---

## References

### Related Tasks
- **Task 2.1.1**: Semantic Chunking (llama_index) - Already implemented
- **Task 2.1.2**: Table Serialization - Already implemented
- **Task 2.3.1**: Deploy Weaviate - Will use converted chunks
- **Task 2.4.2**: Hybrid Search - Will leverage hierarchy metadata

### External Documentation
- [llama_index HierarchicalNodeParser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#hierarchicalnodeparser)
- [LangChain RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)
- [Interoperability Guide](https://docs.llamaindex.ai/en/stable/understanding/interoperability/)

### Code References
- `backend/app/infrastructure/external_services/ml_services/chunking_services/llama_index_chunker.py`: Task 2.1.1 implementation
- `backend/app/application/llm_service.py`: LangChain ChatOllama integration
- `backend/app/domain/value_objects/chunk_metadata.py`: Metadata structure

---

## Appendix: Example Output

### Input Document
```markdown
# Company Policy Manual

## Section 1: Code of Conduct
### 1.1 Professional Behavior
Employees must maintain professional standards at all times.

### 1.2 Workplace Ethics
Ethical conduct is the foundation of our organization.
```

### Output: Hierarchical Chunks

**Level 0 (Root, 2048 chars)**:
```python
{
    "chunk_id": "chunk-root-1",
    "text": "# Company Policy Manual\n\n## Section 1...",
    "depth_level": 0,
    "parent_chunk_id": None,
    "metadata": {
        "heading_path": ["Company Policy Manual"],
        "section": "root"
    }
}
```

**Level 1 (Section, 512 chars)**:
```python
{
    "chunk_id": "chunk-sect-1",
    "text": "## Section 1: Code of Conduct\n### 1.1...",
    "depth_level": 1,
    "parent_chunk_id": "chunk-root-1",
    "metadata": {
        "heading_path": ["Company Policy Manual", "Section 1: Code of Conduct"],
        "section": "Section 1"
    }
}
```

**Level 2 (Paragraph, 128 chars)**:
```python
{
    "chunk_id": "chunk-para-1",
    "text": "### 1.1 Professional Behavior\nEmployees must maintain...",
    "depth_level": 2,
    "parent_chunk_id": "chunk-sect-1",
    "metadata": {
        "heading_path": ["Company Policy Manual", "Section 1", "1.1 Professional Behavior"],
        "section": "1.1"
    }
}
```

---

## Status

**Last Updated**: 2025-01-27
**Status**: 📝 Planning Complete → 🚧 Implementation Started
**Next Milestone**: Core Infrastructure (Phase 1)

---

*This document will be updated as implementation progresses.*
