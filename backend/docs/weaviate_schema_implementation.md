# Weaviate Schema Implementation (Task 2.3.2)

**Status**: ✅ Complete
**Version**: 2.3.2
**Date**: 2025-01-28

## Overview

This document describes the implementation of the Weaviate schema for document chunks with comprehensive metadata support for the RAG system.

## Schema Design

### Class: `Chunk`

The `Chunk` class stores document chunks with embeddings and rich metadata for filtering and retrieval.

**Vector Configuration**:
- Vectorizer: `none` (we provide pre-computed embeddings)
- Index Type: HNSW (Hierarchical Navigable Small World)
- Distance Metric: Cosine similarity
- Dimension: 384 (all-MiniLM-L6-v2 model)

### Property Categories

#### 1. Core Properties (4 fields)
| Property | Type | Description | Filterable | Searchable |
|----------|------|-------------|------------|------------|
| `text` | text | Chunk text content | ✅ | ✅ |
| `chunk_id` | uuid | Unique chunk identifier | ✅ | ❌ |
| `document_id` | uuid | Source document identifier | ✅ | ❌ |
| `sequence` | int | Chunk sequence in document | ✅ (range) | ❌ |

#### 2. Model Metadata (2 fields)
| Property | Type | Description | Filterable |
|----------|------|-------------|------------|
| `model_name` | string | Embedding model name | ✅ |
| `model_version` | string | Embedding model version | ✅ |

#### 3. Document Metadata (8 fields)
| Property | Type | Description | Filterable | Use Case |
|----------|------|-------------|------------|----------|
| `doc_type` | string | Document type (pdf, docx, txt, md) | ✅ | Filter by file type |
| `source_type` | string | Source type (chunk, query) | ✅ | Distinguish chunks vs queries |
| `file_name` | string | Source file name | ✅ | Find specific documents |
| `section` | string | Document section/chapter | ✅ | Hierarchical navigation |
| `category` | string | Document category | ✅ | Categorize content |
| `department` | string | Department/team | ✅ | Access control |
| `author` | string | Document author/creator | ✅ | Attribution |
| `language` | string | Content language (en, es, fr) | ✅ | Multilingual support |

#### 4. Hierarchical Context (2 fields)
| Property | Type | Description | Filterable | Use Case |
|----------|------|-------------|------------|----------|
| `parent_chunk_id` | uuid | Parent chunk ID | ✅ | Hierarchical retrieval |
| `tags` | string[] | Tags for categorization | ✅ | Flexible tagging |

#### 5. Quality Metrics (3 fields)
| Property | Type | Description | Filterable | Use Case |
|----------|------|-------------|------------|----------|
| `token_count` | int | Token count in chunk | ✅ (range) | Filter by size |
| `confidence_score` | number | Chunking confidence (0-1) | ✅ (range) | Quality filtering |
| `quality_score` | number | Content quality (0-1) | ✅ (range) | Quality filtering |

#### 6. Timestamps (2 fields)
| Property | Type | Description | Filterable |
|----------|------|-------------|------------|
| `created_at` | date | Creation timestamp | ✅ (range) |
| `updated_at` | date | Last update timestamp | ✅ (range) |

**Total**: 21 properties

## Implementation Files

### 1. Schema Definition
**File**: `app/infrastructure/external_services/vector_stores/weaviate_schema.py`

Key functions:
```python
# Create schema definition
schema = create_chunk_class_schema(
    dimension=384,
    distance_metric="cosine",
    description="Document chunks with embeddings"
)

# Validate metadata
validate_chunk_metadata(metadata)

# Create metadata template
metadata = create_chunk_metadata_template(
    chunk_id=uuid4(),
    document_id=uuid4(),
    text="Sample chunk",
    doc_type="pdf",
    department="engineering"
)

# Get filterable/searchable fields
filterable = get_filterable_fields()  # 20 fields
searchable = get_searchable_fields()  # 1 field (text)
```

### 2. Configuration
**File**: `app/config/vector_stores/weaviate_config.py`

Configuration classes:
- `WeaviateConfig`: Connection and client settings
- `WeaviateAuthType`: Authentication types (none, api_key, oidc)
- `ConsistencyLevel`: Write consistency (ONE, QUORUM, ALL)

### 3. Initialization Script
**File**: `scripts/setup_weaviate_schema.py`

Usage:
```bash
# Initialize schema (skip if exists)
python scripts/setup_weaviate_schema.py

# Force recreate schema
python scripts/setup_weaviate_schema.py --force
```

### 4. Tests
**File**: `tests/unit/vector_stores/test_weaviate_schema.py`

Test coverage:
- ✅ Schema structure (21 properties)
- ✅ Field types and configurations
- ✅ HNSW index settings
- ✅ Metadata validation
- ✅ Template creation
- ✅ Utility functions

## HNSW Index Configuration

Optimized for performance with 1M+ vectors:

```python
"vectorIndexConfig": {
    "distance": "cosine",           # Cosine similarity
    "ef": -1,                        # Dynamic ef (auto-tuned)
    "efConstruction": 128,           # Build-time exploration
    "maxConnections": 64,            # Graph connectivity
    "vectorCacheMaxObjects": 1000000 # Cache 1M vectors
}
```

**Performance characteristics**:
- Sub-200ms similarity search (k=10)
- Scales to 1M+ vectors
- Optimized for 384-dimensional embeddings
- Trade-off: Build time vs search speed

## Usage Examples

### 1. Basic Chunk Insertion

```python
from uuid import uuid4
from app.infrastructure.external_services.vector_stores.weaviate_schema import (
    create_chunk_metadata_template
)

# Create metadata
metadata = create_chunk_metadata_template(
    chunk_id=uuid4(),
    document_id=uuid4(),
    text="Sample chunk text content",
    sequence=0,
    doc_type="pdf",
    file_name="training_guide.pdf",
    section="Introduction",
    department="engineering",
    tags=["training", "onboarding"]
)

# Insert with embedding
weaviate_client.data_object.create(
    data_object=metadata,
    class_name="Chunk",
    vector=embedding_vector
)
```

### 2. Filtered Search

```python
# Search with department filter
results = weaviate_client.query.get(
    "Chunk", ["text", "file_name", "section"]
).with_near_vector({
    "vector": query_embedding
}).with_where({
    "path": ["department"],
    "operator": "Equal",
    "valueString": "engineering"
}).with_limit(5).do()
```

### 3. Date Range Query

```python
# Find recent chunks
from datetime import datetime, timedelta

week_ago = datetime.utcnow() - timedelta(days=7)

results = weaviate_client.query.get(
    "Chunk", ["text", "created_at"]
).with_where({
    "path": ["created_at"],
    "operator": "GreaterThan",
    "valueDate": week_ago.isoformat()
}).do()
```

### 4. Multi-Filter Query

```python
# Search with multiple filters
results = weaviate_client.query.get(
    "Chunk", ["text", "quality_score"]
).with_near_vector({
    "vector": query_embedding
}).with_where({
    "operator": "And",
    "operands": [
        {
            "path": ["doc_type"],
            "operator": "Equal",
            "valueString": "pdf"
        },
        {
            "path": ["quality_score"],
            "operator": "GreaterThan",
            "valueNumber": 0.8
        }
    ]
}).with_limit(10).do()
```

## Validation Rules

### Required Fields
- `chunk_id`: Valid UUID
- `document_id`: Valid UUID
- `text`: Non-empty string

### Optional Field Constraints
- `sequence`: Non-negative integer
- `token_count`: Non-negative integer
- `confidence_score`: Float in [0, 1]
- `quality_score`: Float in [0, 1]
- `tags`: Array of strings
- All string fields: Must be strings
- All UUID fields: Must be valid UUIDs
- All date fields: Must be datetime or ISO string

## Migration Strategy

### From Basic to Enhanced Schema

If upgrading from basic schema (12 fields → 21 fields):

1. **Backup existing data**:
   ```bash
   # Export existing chunks
   weaviate-cli backup create --class Chunk
   ```

2. **Delete old schema**:
   ```bash
   python scripts/setup_weaviate_schema.py --force
   ```

3. **Re-ingest with enhanced metadata**:
   - Update ingestion pipeline to include new fields
   - Populate department, category, tags, etc.
   - Set default values for quality scores

4. **Verify migration**:
   ```python
   # Check schema
   class_def = weaviate_client.schema.get("Chunk")
   assert len(class_def["properties"]) == 21
   ```

## Performance Considerations

### Indexing Strategy
- **All metadata fields**: Filterable for flexible queries
- **Text field only**: Searchable for full-text search
- **Range filters**: Enabled for numeric and date fields

### Query Optimization
1. **Use specific filters**: Narrow results before vector search
2. **Leverage tags**: Flexible categorization without schema changes
3. **Quality scores**: Pre-filter low-quality chunks
4. **Date ranges**: Efficient temporal queries

### Storage Efficiency
- Optional fields stored as `null` if not provided
- Arrays (tags) stored efficiently in Weaviate
- UUIDs stored natively (not strings)

## Next Steps (Task 2.3.3)

With schema complete, proceed to:

1. **Ingestion Pipeline**: Connect chunks → embeddings → Weaviate
2. **Batch Processing**: Implement bulk insertion (10K chunks/hour)
3. **Error Handling**: Retry logic and deduplication
4. **Monitoring**: Track ingestion metrics and errors

## References

- **Weaviate Docs**: https://weaviate.io/developers/weaviate
- **HNSW Algorithm**: https://arxiv.org/abs/1603.09320
- **Schema Best Practices**: https://weaviate.io/developers/weaviate/config-refs/schema
- **Task 2.2.1**: Embedding model implementation (complete)
- **Task 2.3.3**: Ingestion pipeline (next)
