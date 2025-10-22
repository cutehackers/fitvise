# RAG

## **Data pipeline**

### **Ingestion**

- **initialize**
  - audit sources
  - categorize sources

- **discover**
  - files
  - database connectors and capture samples
  - web scraping
  - document external APIs
  
- **process**
  - extract raw content (docling, tika)
  - normalize text tokens and structure (spacy)
  - enrich metadata (spacy)
  - validate content quality / scores (great_expectations)
  - persist normalized payloads in object storage and the in-memory repository

- **post-processing**
  - load chunking preset/overrides and attach run metadata
  - execute semantic chunking against processed document IDs
  - capture chunk counts and append recoverable errors without failing the run
  