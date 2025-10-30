# RAG Build Pipeline

Complete modular RAG ingestion pipeline with 3 phases: infrastructure validation, document ingestion, and embedding generation.

## Overview

The RAG Build Pipeline is designed to process documents through a complete ingestion workflow:

1. **Phase 1: Infrastructure Setup and Validation** (`setup_rag_infrastructure.py`)
2. **Phase 2: Document Ingestion and Processing** (`ingestion.py`)
3. **Phase 3: Embedding Generation and Storage** (`embed_pipeline.py`)

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

### Complete Pipeline (Recommended)

Run all phases sequentially with detailed reporting:

```bash
python scripts/build_rag_pipeline.py --config my_rag_pipeline.yaml --verbose --output-dir ./reports
```

Options:
- `--config`: Path to configuration file (required)
- `--output-dir`: Directory to save reports
- `--verbose`: Enable detailed logging
- `--dry-run`: Run without persisting data
- `--batch-size`: Embedding batch size (default: 32)
- `--document-limit`: Limit number of documents to process

### Individual Phases

You can also run each phase separately:

#### Phase 1: Infrastructure Validation
```bash
python scripts/setup_rag_infrastructure.py --config my_rag_pipeline.yaml --verbose
```

#### Phase 2: Document Ingestion
```bash
python scripts/ingestion.py --config my_rag_pipeline.yaml --verbose --output ingestion_summary.json
```

#### Phase 3: Embedding Generation
```bash
python scripts/embed_pipeline.py --config my_rag_pipeline.yaml --verbose --batch-size 32 --output embedding_results.json
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

For detailed debugging, run individual phases with `--verbose`:

```bash
# Debug infrastructure
python scripts/setup_rag_infrastructure.py --config my_rag_pipeline.yaml --verbose

# Debug ingestion with dry run
python scripts/ingestion.py --config my_rag_pipeline.yaml --verbose --dry-run

# Debug embedding with small batch
python scripts/embed_pipeline.py --config my_rag_pipeline.yaml --verbose --batch-size 4 --document-limit 2
```

## Architecture Details

### Phase 1: Infrastructure Validation
- Validates embedding model (`Alibaba-NLP/gte-multilingual-base`)
- Checks Weaviate connection and DocumentChunk schema
- Tests object storage accessibility
- Validates configuration file integrity

### Phase 2: Document Ingestion
- Discovers documents based on configured patterns
- Extracts content using Docling (PDFs) and Tika (other formats)
- Normalizes text and extracts metadata
- Applies semantic chunking (200-500 token chunks)
- Stores processed documents and metadata in shared repository

### Phase 3: Embedding Generation
- Retrieves processed documents from **shared repository** (continuity from Phase 2)
- Generates embeddings using the multilingual model
- Applies SHA256-based deduplication
- Stores embeddings in Weaviate with metadata
- Provides comprehensive statistics and error reporting

## 📋 Repository Architecture Fix

**Problem Solved**: Fixed the critical issue where ingestion and embedding phases used different `InMemoryDocumentRepository` instances, causing embeddings to fail because no documents were found.

**Solution**: Implemented dependency injection from the orchestrator:
- Orchestrator creates shared `InMemoryDocumentRepository` instance
- Phase 2 stores processed documents in shared repository
- Phase 3 retrieves documents from the same shared repository
- Ensures data continuity across pipeline phases

## Integration with Existing Code

The pipeline leverages your existing components:

- `SetupEmbeddingInfrastructureUseCase`: Infrastructure validation
- `run_pipeline()` orchestrator: Document processing
- `BuildIngestionPipelineUseCase`: Embedding generation
- `SentenceTransformerService`: Embedding model
- `WeaviateClient`: Vector storage

This modular design allows you to run the complete pipeline or individual phases as needed, with comprehensive reporting and error handling throughout.