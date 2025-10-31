# RAG Build Pipeline

Complete modular RAG ingestion pipeline with 3 phases: infrastructure validation, document ingestion, and embedding generation.

## Overview

The RAG Build Pipeline is designed to process documents through a complete ingestion workflow with a clean 4-layer architecture that supports dependency injection and independent phase execution:

1. **Phase 1: Infrastructure Setup and Validation** (`InfrastructurePhase`)
2. **Phase 2: Document Ingestion and Processing** (`IngestionPhase`)
3. **Phase 3: Embedding Generation and Storage** (`EmbeddingPhase`)

All phases are coordinated by the `RAGWorkflow` orchestrator, which ensures proper dependency injection and data continuity across phases.

## Architecture

### 4-Layer Design

The pipeline follows a clean separation of concerns with dependency injection:

```
Layer 1: Repository Interfaces (Domain)
├── DocumentRepository (abstract)
└── DataSourceRepository (abstract)

Layer 2: Phase Classes (Pipeline Logic)
├── InfrastructurePhase (app/pipeline/phases/infrastructure_phase.py)
├── IngestionPhase (app/pipeline/phases/ingestion_phase.py)
└── EmbeddingPhase (app/pipeline/phases/embedding_phase.py)

Layer 3: Workflow Orchestrator (Coordination)
└── RAGWorkflow (app/pipeline/workflow.py)
    ├── RepositoryBundle (dependency injection)
    └── Phase coordination with shared repositories

Layer 4: CLI Script (User Interface)
└── build_rag_pipeline.py (single entry point with --phases flag for selective execution)
```

### Key Components

#### RepositoryBundle

Ensures all phases share the same repository instances for data continuity:

```python
@dataclass
class RepositoryBundle:
    document_repository: DocumentRepository
    data_source_repository: DataSourceRepository
```

#### RAGWorkflow

Central orchestrator that:
- Creates or accepts shared repository instances
- Initializes all phases with proper dependency injection
- Provides methods to run phases independently or as a complete pipeline
- Manages result tracking and reporting

```python
workflow = RAGWorkflow(verbose=True)  # Creates default in-memory repositories
# Or with custom repositories:
repositories = RepositoryBundle(
    document_repository=InMemoryDocumentRepository(),
    data_source_repository=InMemoryDataSourceRepository(),
)
workflow = RAGWorkflow(repositories=repositories, verbose=True)
```

#### Phase Classes

Each phase is a standalone class that:
- Accepts dependencies via constructor (dependency injection)
- Provides an `execute()` method with clear input/output contracts
- Can run independently or as part of the complete pipeline
- Returns structured results for tracking and reporting

### Data Continuity

**Critical Feature**: The workflow ensures processed documents flow correctly between phases:

1. **Phase 2 (Ingestion)**: Stores processed documents in `document_repository`
2. **Phase 3 (Embedding)**: Retrieves documents from the **same** `document_repository` instance
3. **Shared State**: Both phases use the same repository instance via dependency injection

This guarantees data continuity throughout pipeline execution with proper dependency injection.

## Prerequisites

Before running the pipeline, ensure the following services are running:

- **Weaviate**: Vector database (default: http://localhost:8080)
- **MinIO** (if using `provider: minio`): Object storage (default: localhost:9000)
- **Python Environment**: All dependencies installed

### Quick Setup with Docker Compose

```bash
# Start required services
docker-compose up -d weaviate minio

# Verify services are running
docker-compose ps
```

## Configuration

Create a configuration file based on the sample:

```bash
# Copy sample configuration
cp rag_pipeline_sample.yaml my_rag_pipeline.yaml

# Edit the configuration
nano my_rag_pipeline.yaml
```

### Key Configuration Sections

- `documents.path`: Directory containing your documents
- `documents.include`: File patterns to process (e.g., `["*.pdf", "*.md"]`)
- `storage.provider`: Storage backend (`local`, `minio`, or `s3`)
- `storage.bucket`: Storage bucket name

## Usage

### Complete Pipeline (Default)

Run all phases sequentially with detailed reporting:

```bash
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --verbose --output-dir ./reports
```

Options:
- `--config`: Path to configuration file (required)
- `--phases`: Specific phases to run (choices: infrastructure, ingestion, embedding). If omitted, all phases run.
- `--output-dir`: Directory to save reports
- `--verbose`: Enable detailed logging
- `--dry-run`: Run without persisting data
- `--batch-size`: Embedding batch size (default: 32)
- `--document-limit`: Limit number of documents to process

### Individual Phases

You can run specific phases using the `--phases` flag or use the RAGWorkflow programmatically:

#### Phase 1: Infrastructure Validation

**Using CLI with --phases flag**:
```bash
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --phases infrastructure --verbose
```

**Using RAGWorkflow** (programmatic):
```python
from app.pipeline.config import PipelineSpec
from app.pipeline.workflow import RAGWorkflow

spec = PipelineSpec.from_file("my_rag_pipeline.yaml")
workflow = RAGWorkflow(verbose=True)
result = await workflow.run_infrastructure_check(spec)

print(f"Success: {result.success}")
print(f"Errors: {result.errors}")
```

#### Phase 2: Document Ingestion

**Using CLI with --phases flag**:
```bash
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --phases ingestion --verbose
```

**Using RAGWorkflow** (programmatic):
```python
from app.pipeline.config import PipelineSpec
from app.pipeline.workflow import RAGWorkflow

spec = PipelineSpec.from_file("my_rag_pipeline.yaml")
workflow = RAGWorkflow(verbose=True)
summary = await workflow.run_ingestion(spec, dry_run=False)

print(f"Processed: {summary.processed}")
print(f"Chunks: {summary.counters.get('chunking', {}).get('total_chunks', 0)}")
```

#### Phase 3: Embedding Generation

**Using CLI with --phases flag**:
```bash
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --phases embedding --verbose --batch-size 32
```

**Using RAGWorkflow** (programmatic):
```python
from app.pipeline.config import PipelineSpec
from app.pipeline.workflow import RAGWorkflow

spec = PipelineSpec.from_file("my_rag_pipeline.yaml")
workflow = RAGWorkflow(verbose=True)
result = await workflow.run_embedding(
    spec,
    batch_size=32,
    document_limit=None,
)

print(f"Embeddings stored: {result.embeddings_stored}")
print(f"Success rate: {result.as_dict()['embedding_success_rate']}%")
```

#### Multiple Phases

You can also run multiple specific phases together:

```bash
# Run infrastructure validation and ingestion only
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --phases infrastructure ingestion --verbose

# Run ingestion and embedding only (skip validation)
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --phases ingestion embedding --verbose
```

### Complete Pipeline with Custom Repositories

For advanced use cases with custom repository implementations:

```python
from app.pipeline.config import PipelineSpec
from app.pipeline.workflow import RAGWorkflow, RepositoryBundle
from app.infrastructure.repositories.in_memory_document_repository import InMemoryDocumentRepository
from app.infrastructure.repositories.in_memory_data_source_repository import InMemoryDataSourceRepository

# Create custom repository instances
repositories = RepositoryBundle(
    document_repository=InMemoryDocumentRepository(),
    data_source_repository=InMemoryDataSourceRepository(),
)

# Initialize workflow with custom repositories
spec = PipelineSpec.from_file("my_rag_pipeline.yaml")
workflow = RAGWorkflow(repositories=repositories, verbose=True)

# Run complete pipeline
summary = await workflow.run_complete_pipeline(
    spec=spec,
    dry_run=False,
    batch_size=32,
    document_limit=None,
    output_dir="./reports",
)

summary.print_summary()
```

## Output Structure

When using `--output-dir`, the pipeline creates:

```
./reports/
├── rag_ingestion_summary.json     # Main summary report
├── rag_ingestion_report.txt       # Human-readable report
├── phase_1_infrastructure.json    # Phase 1 detailed results
├── phase_2_ingestion.json         # Phase 2 detailed results
├── phase_3_embedding.json         # Phase 3 detailed results
├── ingestion_detailed.json         # Complete ingestion summary
└── embedding_detailed.json         # Complete embedding summary
```

## Example Workflow

### 1. Prepare Your Documents

```bash
# Create data directory
mkdir -p data/sample

# Add your fitness documents
cp ~/Documents/fitness/*.pdf data/sample/
cp ~/Documents/workouts/*.md data/sample/
```

### 2. Configure the Pipeline

Edit `my_rag_pipeline.yaml`:

```yaml
documents:
  path: ./data/sample
  recurse: true
  include: ["*.pdf", "*.md"]

storage:
  provider: local
  base_dir: ./storage
  bucket: rag-source
```

### 3. Run the Complete Pipeline

```bash
python scripts/build_rag_pipeline.py \
  --config my_rag_pipeline.yaml \
  --verbose \
  --output-dir ./reports
```

### 4. Review Results

The pipeline will print a summary like:

```
🚀 RAG Ingestion Pipeline - EXECUTION SUMMARY
===============================================================================
Status: ✅ SUCCESS
Execution Time: 2m 15.30s
Phases Completed: 3/3
Total Errors: 0
Total Warnings: 1

📊 AGGREGATED METRICS:
Documents Processed: 15
Chunks Generated: 127
Embeddings Stored: 125
Average Chunks per Document: 8.47
Embedding Success Rate: 98.43%

📋 PHASE RESULTS:
  ✅ Phase Infrastructure Setup: 3.21s
  ✅ Phase Document Ingestion: 45.67s
  ✅ Phase Embedding Generation: 86.42s
```

## Troubleshooting

### Common Issues

#### 1. Weaviate Connection Failed
```bash
❌ Critical: Weaviate connection failed
```

**Solution**: Ensure Weaviate is running:
```bash
docker-compose up -d weaviate
# Check status
curl http://localhost:8080/v1/.well-known/ready
```

#### 2. No Documents Found
```bash
⚠️ No files found with specified patterns
```

**Solution**: Check your configuration and document paths:
```bash
# Verify directory exists
ls -la ./data/sample/

# Check file patterns
find ./data/sample -name "*.pdf" -o -name "*.md"
```

#### 3. Embedding Model Loading Failed
```bash
Critical: Embedding service setup failed
```

**Solution**: Ensure you have sufficient memory and internet connection for model download:
```bash
# Check available memory
free -h

# Test model loading manually (if needed)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Alibaba-NLP/gte-multilingual-base')"
```

#### 4. Storage Bucket Issues
```bash
Critical: Storage bucket not accessible
```

**Solution**: For MinIO, check credentials and bucket:
```bash
# Check MinIO is running
docker-compose ps minio

# Test bucket access (if using MinIO)
mc ls local/
```

### Debug Mode

For detailed debugging, run individual phases with `--verbose` and `--phases`:

```bash
# Debug infrastructure
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --phases infrastructure --verbose

# Debug ingestion with dry run
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --phases ingestion --verbose --dry-run

# Debug embedding with small batch
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --phases embedding --verbose --batch-size 4 --document-limit 2
```

## Phase Implementation Details

### Phase 1: InfrastructurePhase (`app/pipeline/phases/infrastructure_phase.py`)

**Responsibilities**:
- Validates embedding model (`Alibaba-NLP/gte-multilingual-base`)
- Checks Weaviate connection and DocumentChunk schema
- Tests object storage accessibility
- Validates configuration file integrity

**Key Method**:
```python
async def execute(self, spec: PipelineSpec) -> InfrastructureValidationResult:
    """Execute infrastructure validation phase."""
    # Returns structured validation results
```

**Dependencies**: None (stateless validation)

### Phase 2: IngestionPhase (`app/pipeline/phases/ingestion_phase.py`)

**Responsibilities**:
- Discovers documents based on configured patterns
- Extracts content using Docling (PDFs) and Tika (other formats)
- Normalizes text and extracts metadata
- Applies semantic chunking (200-500 token chunks)
- Stores processed documents and metadata in **shared repository**

**Key Method**:
```python
async def execute(self, spec: PipelineSpec, dry_run: bool = False) -> RunSummary:
    """Execute document ingestion phase."""
    # Uses injected document_repository to store results
```

**Dependencies** (via constructor):
- `document_repository: DocumentRepository` - **Shared instance for data continuity**
- `data_source_repository: Optional[DataSourceRepository]`

### Phase 3: EmbeddingPhase (`app/pipeline/phases/embedding_phase.py`)

**Responsibilities**:
- Retrieves processed documents from **shared repository** (continuity from Phase 2)
- Generates embeddings using the multilingual model
- Applies SHA256-based deduplication
- Stores embeddings in Weaviate with metadata
- Provides comprehensive statistics and error reporting

**Key Method**:
```python
async def execute(
    self,
    spec: PipelineSpec,
    batch_size: int = 32,
    deduplication_enabled: bool = True,
    max_retries: int = 3,
    show_progress: bool = True,
    document_limit: Optional[int] = None,
) -> EmbeddingResult:
    """Execute embedding generation phase."""
    # Uses injected document_repository to retrieve documents
```

**Dependencies** (via constructor):
- `document_repository: DocumentRepository` - **Same instance from Phase 2**

## Dependency Injection Architecture

### RepositoryBundle Pattern

**Implementation**:
1. **RepositoryBundle** dataclass bundles shared repository instances
2. **RAGWorkflow** creates or accepts repository bundle in constructor
3. **Phase classes** receive repositories via constructor (dependency injection)
4. **All phases** use the same repository instances throughout pipeline execution

**Code Flow**:
```python
# Workflow creates shared repositories
repositories = RepositoryBundle(
    document_repository=InMemoryDocumentRepository(),  # Single instance
    data_source_repository=InMemoryDataSourceRepository(),
)

# Workflow injects into phase classes
self.ingestion_phase = IngestionPhase(
    document_repository=self.repositories.document_repository,  # Shared
    data_source_repository=self.repositories.data_source_repository,
)
self.embedding_phase = EmbeddingPhase(
    document_repository=self.repositories.document_repository,  # Same instance!
)

# Data flows correctly:
# Phase 2 stores → shared repository → Phase 3 retrieves
```

**Benefits**:
- ✅ Data continuity across all phases
- ✅ Testable with mock repositories
- ✅ Support for both in-memory and persistent repositories
- ✅ Clear dependency graph
- ✅ No hidden state or global variables

## Final Architecture

### Clean, Consolidated Architecture

```
scripts/
└── build_rag_pipeline.py (203 lines)
    └── Single entry point with --phases flag

app/pipeline/
├── config.py (configuration models)
├── contracts.py (data contracts)
├── workflow.py (orchestration)
└── phases/
    ├── infrastructure_phase.py (self-contained)
    ├── ingestion_phase.py (1086 lines - fully self-contained!)
    └── embedding_phase.py (self-contained)
```

**Benefits**:
- ✅ **Self-contained phases**: All logic in phase classes, no external dependencies
- ✅ **Single CLI entry point**: One script with flexible `--phases` flag
- ✅ **No extra layers**: Direct phase → workflow relationship
- ✅ **Clean architecture**: Clear responsibility boundaries
- ✅ **Production ready**: Clean architecture suitable for launch


## Quality Improvements

### Architecture Quality
- ✅ **No External Orchestrator**: Phase classes are truly self-contained
- ✅ **Single Entry Point**: One CLI script instead of four
- ✅ **Flexible Execution**: Run any combination of phases via `--phases` flag
- ✅ **Clear Boundaries**: No confusing phase → orchestrator → implementation flow
- ✅ **Production Ready**: Clean architecture suitable for launch

### Code Organization
- ✅ **Logical Grouping**: All ingestion logic in `IngestionPhase` class
- ✅ **Instance Methods**: Helper functions converted to instance methods for clarity
- ✅ **No Globals**: All state managed through dependency injection
- ✅ **Consistent Pattern**: All three phases follow same self-contained pattern

### Developer Experience
- ✅ **Simpler Codebase**: One place to look for ingestion logic
- ✅ **Easier Testing**: Can test `IngestionPhase` directly without orchestrator
- ✅ **Better Documentation**: README reflects actual architecture
- ✅ **Clear Migration Path**: Straightforward command updates

## Integration with Existing Code

The pipeline leverages your existing components:

**Domain Layer**:
- `DocumentRepository`: Abstract interface for document storage
- `DataSourceRepository`: Abstract interface for data source metadata

**Infrastructure Layer**:
- `InMemoryDocumentRepository`: In-memory implementation (default)
- `InMemoryDataSourceRepository`: In-memory implementation (default)

**Use Cases**:
- `SetupEmbeddingInfrastructureUseCase`: Infrastructure validation
- `SemanticChunkingUseCase`: Document chunking
- `BuildIngestionPipelineUseCase`: Embedding generation
- `run_pipeline()` orchestrator: Document processing

**Services**:
- `SentenceTransformerService`: Embedding model
- `WeaviateClient`: Vector storage

**Workflow Integration**:
- `InfrastructurePhase` → delegates to `SetupEmbeddingInfrastructureUseCase`
- `IngestionPhase` → delegates to `run_pipeline()` with injected repositories
- `EmbeddingPhase` → delegates to `BuildIngestionPipelineUseCase` with injected repositories

This modular design allows you to run the complete pipeline or individual phases as needed, with comprehensive reporting and error handling throughout.