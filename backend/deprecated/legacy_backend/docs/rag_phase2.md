> Superseded: use `backend/botadvisor/docs/*` as the canonical source of truth.
> This file is kept only as historical migration reference.

rag-system-phase2/                        # PHASE 2: Indexing and Retrieval System
├── README.md                              # Phase 2 specific documentation
├── requirements-phase2.txt                # Phase 2 dependencies (adds Weaviate, Elasticsearch, ML libs)
├── docker-compose-phase2.yml             # Phase 2 services (+ Weaviate, Elasticsearch)
├── .env.phase2.example
│
├── src/
│   ├── init.py
│   │
│   ├── domain/                            # Domain Layer - Phase 2 Extensions
│   │   ├── init.py
│   │   ├── entities/
│   │   │   ├── init.py
│   │   │   ├── document.py               # Extended from Phase 1
│   │   │   ├── chunk.py                  # NEW: Text chunk entity
│   │   │   ├── embedding.py              # NEW: Vector embedding entity
│   │   │   ├── search_query.py           # NEW: Search query entity
│   │   │   ├── search_result.py          # NEW: Search result entity
│   │   │   ├── index.py                  # NEW: Search index entity
│   │   │   └── data_source.py            # From Phase 1
│   │   │
│   │   ├── value_objects/
│   │   │   ├── init.py
│   │   │   ├── document_metadata.py      # From Phase 1
│   │   │   ├── chunk_metadata.py         # NEW: Chunk metadata VO
│   │   │   ├── embedding_vector.py       # NEW: Vector representation VO
│   │   │   ├── search_filters.py         # NEW: Search criteria VO
│   │   │   ├── similarity_score.py       # NEW: Similarity scoring VO
│   │   │   └── retrieval_context.py      # NEW: Retrieval context VO
│   │   │
│   │   ├── repositories/                 # Repository Interfaces for Phase 2
│   │   │   ├── init.py
│   │   │   ├── document_repository.py    # Extended from Phase 1
│   │   │   ├── chunk_repository.py       # NEW: Chunk storage interface
│   │   │   ├── embedding_repository.py   # NEW: Vector storage interface
│   │   │   ├── search_repository.py      # NEW: Search interface
│   │   │   └── index_repository.py       # NEW: Index management interface
│   │   │
│   │   ├── services/                     # Domain Services for Phase 2
│   │   │   ├── init.py
│   │   │   ├── document_processor.py     # Extended from Phase 1
│   │   │   ├── chunking_service.py       # NEW: Text chunking logic
│   │   │   ├── embedding_service.py      # NEW: Embedding generation logic
│   │   │   ├── retrieval_service.py      # NEW: Search and retrieval logic
│   │   │   ├── ranking_service.py        # NEW: Result ranking logic
│   │   │   └── index_service.py          # NEW: Index management logic
│   │   │
│   │   └── exceptions/
│   │       ├── init.py
│   │       ├── document_exceptions.py    # From Phase 1
│   │       ├── chunking_exceptions.py    # NEW: Chunking errors
│   │       ├── embedding_exceptions.py   # NEW: Embedding errors
│   │       ├── retrieval_exceptions.py   # NEW: Retrieval errors
│   │       └── indexing_exceptions.py    # NEW: Indexing errors
│   │
│   ├── application/                       # Application Layer - Phase 2 Use Cases
│   │   ├── init.py
│   │   ├── use_cases/
│   │   │   ├── init.py
│   │   │   │
│   │   │   ├── data_ingestion/           # From Phase 1 (minimal changes)
│   │   │   │   ├── init.py
│   │   │   │   ├── ingest_documents.py
│   │   │   │   └── process_documents.py
│   │   │   │
│   │   │   ├── chunking/                 # Epic 2.1: Chunking and Text Processing
│   │   │   │   ├── init.py
│   │   │   │   ├── semantic_chunking.py       # Task 2.1.1
│   │   │   │   ├── table_serialization.py     # Task 2.1.2
│   │   │   │   └── recursive_chunking.py      # Task 2.1.3
│   │   │   │
│   │   │   ├── embedding/                # Epic 2.2: Embedding Model Pipeline
│   │   │   │   ├── init.py
│   │   │   │   ├── setup_embedding_infrastructure.py  # Task 2.2.1
│   │   │   │   ├── fine_tune_embeddings.py            # Task 2.2.2
│   │   │   │   ├── optimize_inference.py              # Task 2.2.3
│   │   │   │   └── multimodal_embeddings.py           # Task 2.2.4
│   │   │   │
│   │   │   ├── indexing/                 # Epic 2.3: Vector Database Setup
│   │   │   │   ├── init.py
│   │   │   │   ├── deploy_vector_db.py            # Task 2.3.1
│   │   │   │   ├── design_schema.py               # Task 2.3.2
│   │   │   │   └── build_ingestion_pipeline.py    # Task 2.3.3
│   │   │   │
│   │   │   └── retrieval/                # Epic 2.4: Hybrid Retrieval System
│   │   │       ├── init.py
│   │   │       ├── keyword_search.py             # Task 2.4.1
│   │   │       ├── hybrid_search.py              # Task 2.4.2
│   │   │       ├── rerank_results.py             # Task 2.4.3
│   │   │       └── query_classification.py       # Task 2.4.4
│   │   │
│   │   ├── dto/                          # Data Transfer Objects
│   │   │   ├── init.py
│   │   │   ├── document_dto.py           # From Phase 1
│   │   │   ├── chunk_dto.py              # NEW: Chunk data transfer
│   │   │   ├── embedding_dto.py          # NEW: Embedding data transfer
│   │   │   ├── search_dto.py             # NEW: Search request/response
│   │   │   └── index_dto.py              # NEW: Index management
│   │   │
│   │   └── interfaces/                   # Application Interfaces
│   │       ├── init.py
│   │       ├── document_processor_interface.py  # From Phase 1
│   │       ├── chunking_interface.py             # NEW: Chunking interface
│   │       ├── embedding_generator_interface.py # NEW: Embedding interface
│   │       ├── vector_store_interface.py         # NEW: Vector storage interface
│   │       ├── search_engine_interface.py        # NEW: Search interface
│   │       └── reranker_interface.py             # NEW: Re-ranking interface
│   │
│   ├── infrastructure/                    # Infrastructure Layer - Phase 2
│   │   ├── init.py
│   │   ├── persistence/
│   │   │   ├── init.py
│   │   │   ├── repositories/             # Extended repository implementations
│   │   │   │   ├── init.py
│   │   │   │   ├── postgres_document_repository.py    # From Phase 1
│   │   │   │   ├── postgres_chunk_repository.py       # NEW: Chunk storage
│   │   │   │   ├── weaviate_embedding_repository.py   # NEW: Vector storage
│   │   │   │   ├── elasticsearch_search_repository.py # NEW: Keyword search
│   │   │   │   └── hybrid_search_repository.py        # NEW: Combined search
│   │   │   │
│   │   │   ├── models/                   # Database Models
│   │   │   │   ├── init.py
│   │   │   │   ├── document_model.py     # From Phase 1
│   │   │   │   ├── chunk_model.py        # NEW: Chunk model
│   │   │   │   ├── embedding_model.py    # NEW: Embedding metadata model
│   │   │   │   └── search_index_model.py # NEW: Index model
│   │   │   │
│   │   │   └── migrations/
│   │   │       ├── init.py
│   │   │       ├── 001_initial_tables.py     # From Phase 1
│   │   │       ├── 002_add_metadata_fields.py # From Phase 1
│   │   │       ├── 003_add_chunk_tables.py    # NEW: Chunk tables
│   │   │       └── 004_add_embedding_tables.py # NEW: Embedding metadata
│   │   │
│   │   ├── external_services/
│   │   │   ├── init.py
│   │   │   ├── data_sources/             # From Phase 1 (minimal changes)
│   │   │   │   ├── init.py
│   │   │   │   ├── database_connectors/
│   │   │   │   ├── web_scrapers/
│   │   │   │   └── file_processors/
│   │   │   │
│   │   │   ├── ml_services/              # NEW: ML Service Integrations
│   │   │   │   ├── init.py
│   │   │   │   ├── embedding_models/     # Epic 2.2: Embedding Services
│   │   │   │   │   ├── init.py
│   │   │   │   │   ├── sentence_transformer_service.py  # Task 2.2.1
│   │   │   │   │   ├── huggingface_embedding_service.py
│   │   │   │   │   ├── openai_embedding_service.py
│   │   │   │   │   ├── onnx_embedding_service.py        # Task 2.2.3
│   │   │   │   │   └── clip_embedding_service.py        # Task 2.2.4
│   │   │   │   │
│   │   │   │   ├── chunking_services/    # Epic 2.1: Chunking Services
│   │   │   │   │   ├── init.py
│   │   │   │   │   ├── llama_index_chunker.py    # Task 2.1.1
│   │   │   │   │   ├── langchain_chunker.py      # Task 2.1.3
│   │   │   │   │   ├── table_processor.py        # Task 2.1.2
│   │   │   │   │   └── recursive_chunker.py
│   │   │   │   │
│   │   │   │   └── reranking_services/   # Epic 2.4: Re-ranking Services
│   │   │   │       ├── init.py
│   │   │   │       ├── cross_encoder_reranker.py  # Task 2.4.3
│   │   │   │       ├── bert_reranker.py
│   │   │   │       └── custom_reranker.py
│   │   │   │
│   │   │   ├── vector_stores/            # Epic 2.3: Vector Database Integrations
│   │   │   │   ├── init.py
│   │   │   │   ├── weaviate_client.py           # Task 2.3.1, 2.3.2, 2.3.3
│   │   │   │   ├── pinecone_client.py
│   │   │   │   ├── chroma_client.py
│   │   │   │   ├── qdrant_client.py
│   │   │   │   └── base_vector_store.py
│   │   │   │
│   │   │   ├── search_engines/           # Epic 2.4: Search Engine Integrations
│   │   │   │   ├── init.py
│   │   │   │   ├── elasticsearch_client.py     # Task 2.4.1
│   │   │   │   ├── opensearch_client.py
│   │   │   │   ├── haystack_client.py          # Task 2.4.2
│   │   │   │   └── base_search_engine.py
│   │   │   │
│   │   │   └── query_classification/     # Epic 2.4: Query Classification
│   │   │       ├── init.py
│   │   │       ├── bart_classifier.py          # Task 2.4.4
│   │   │       ├── zero_shot_classifier.py
│   │   │       └── rule_based_classifier.py
│   │   │
│   │   ├── orchestration/                # Extended Airflow DAGs for Phase 2
│   │   │   ├── init.py
│   │   │   ├── dags/
│   │   │   │   ├── init.py
│   │   │   │   ├── data_ingestion_dag.py       # From Phase 1
│   │   │   │   ├── chunking_dag.py             # NEW: Text chunking pipeline
│   │   │   │   ├── embedding_generation_dag.py # NEW: Embedding pipeline
│   │   │   │   ├── index_building_dag.py       # NEW: Index creation pipeline
│   │   │   │   └── search_index_update_dag.py  # NEW: Index maintenance
│   │   │   │
│   │   │   └── operators/
│   │   │       ├── init.py
│   │   │       ├── document_processor_operator.py  # From Phase 1
│   │   │       ├── chunking_operator.py            # NEW: Chunking operator
│   │   │       ├── embedding_operator.py           # NEW: Embedding operator
│   │   │       ├── vector_store_operator.py        # NEW: Vector storage operator
│   │   │       └── search_index_operator.py        # NEW: Search indexing operator
│   │   │
│   │   ├── storage/                      # Extended storage for Phase 2
│   │   │   ├── init.py
│   │   │   ├── object_storage/
│   │   │   │   ├── init.py
│   │   │   │   ├── minio_client.py       # From Phase 1
│   │   │   │   ├── embedding_storage.py  # NEW: Embedding file storage
│   │   │   │   └── chunk_storage.py      # NEW: Processed chunk storage
│   │   │   │
│   │   │   └── cache/
│   │   │       ├── init.py
│   │   │       ├── redis_client.py       # From Phase 1
│   │   │       ├── embedding_cache.py    # NEW: Embedding caching
│   │   │       └── search_cache.py       # NEW: Search result caching
│   │   │
│   │   └── monitoring/                   # NEW: Advanced monitoring for Phase 2
│   │       ├── init.py
│   │       ├── metrics/
│   │       │   ├── init.py
│   │       │   ├── embedding_metrics.py
│   │       │   ├── search_metrics.py
│   │       │   ├── performance_metrics.py
│   │       │   └── quality_metrics.py
│   │       │
│   │       └── logging/
│   │           ├── init.py
│   │           ├── search_logger.py
│   │           ├── embedding_logger.py
│   │           └── performance_logger.py
│   │
│   └── presentation/                     # Extended Presentation for Phase 2
│       ├── init.py
│       ├── api/                          # Extended API for search functionality
│       │   ├── init.py
│       │   ├── v1/
│       │   │   ├── init.py
│       │   │   ├── endpoints/
│       │   │   │   ├── init.py
│       │   │   │   ├── documents.py      # From Phase 1
│       │   │   │   ├── chunks.py         # NEW: Chunk management
│       │   │   │   ├── embeddings.py     # NEW: Embedding management
│       │   │   │   ├── search.py         # NEW: Search endpoints
│       │   │   │   ├── indices.py        # NEW: Index management
│       │   │   │   └── health.py         # Extended health check
│       │   │   │
│       │   │   └── schemas/
│       │   │       ├── init.py
│       │   │       ├── document_schemas.py  # From Phase 1
│       │   │       ├── chunk_schemas.py     # NEW: Chunk schemas
│       │   │       ├── embedding_schemas.py # NEW: Embedding schemas
│       │   │       ├── search_schemas.py    # NEW: Search schemas
│       │   │       └── index_schemas.py     # NEW: Index schemas
│       │   │
│       │   └── middleware/
│       │       ├── init.py
│       │       ├── logging_middleware.py     # From Phase 1
│       │       ├── search_middleware.py      # NEW: Search-specific middleware
│       │       └── caching_middleware.py     # NEW: Response caching
│       │
│       └── cli/                          # Extended CLI for Phase 2 operations
│           ├── init.py
│           ├── commands/
│           │   ├── init.py
│           │   ├── ingest_command.py         # From Phase 1
│           │   ├── chunk_command.py          # NEW: Chunking operations
│           │   ├── embed_command.py          # NEW: Embedding operations
│           │   ├── index_command.py          # NEW: Index operations
│           │   ├── search_command.py         # NEW: Search testing
│           │   └── benchmark_command.py      # NEW: Performance benchmarking
│           │
│           └── main.py
│
├── tests/                                # Phase 2 Tests
│   ├── init.py
│   ├── unit/
│   │   ├── init.py
│   │   ├── domain/
│   │   │   ├── test_chunk_entity.py
│   │   │   ├── test_embedding_entity.py
│   │   │   ├── test_chunking_service.py
│   │   │   ├── test_embedding_service.py
│   │   │   ├── test_retrieval_service.py
│   │   │   └── test_ranking_service.py
│   │   │
│   │   ├── application/
│   │   │   ├── test_semantic_chunking.py
│   │   │   ├── test_embedding_generation.py
│   │   │   ├── test_vector_storage.py
│   │   │   ├── test_hybrid_search.py
│   │   │   └── test_result_reranking.py
│   │   │
│   │   └── infrastructure/
│   │       ├── test_weaviate_client.py
│   │       ├── test_elasticsearch_client.py
│   │       ├── test_embedding_models.py
│   │       ├── test_chunking_services.py
│   │       └── test_reranking_services.py
│   │
│   ├── integration/
│   │   ├── init.py
│   │   ├── test_chunking_pipeline.py
│   │   ├── test_embedding_pipeline.py
│   │   ├── test_indexing_pipeline.py
│   │   ├── test_search_pipeline.py
│   │   └── test_end_to_end_retrieval.py
│   │
│   ├── performance/                      # NEW: Performance tests
│   │   ├── init.py
│   │   ├── test_embedding_speed.py
│   │   ├── test_search_latency.py
│   │   ├── test_throughput.py
│   │   └── test_memory_usage.py
│   │
│   ├── fixtures/
│   │   ├── sample_documents/             # From Phase 1
│   │   ├── sample_chunks/                # NEW: Test chunks
│   │   ├── sample_embeddings/            # NEW: Test embeddings
│   │   ├── sample_queries/               # NEW: Test search queries
│   │   └── benchmark_data/               # NEW: Performance test data
│   │
│   └── conftest.py
│
├── deployment/                           # Phase 2 Deployment
│   ├── docker/
│   │   ├── Dockerfile.processor          # From Phase 1
│   │   ├── Dockerfile.embedder           # NEW: Embedding service
│   │   ├── Dockerfile.searcher           # NEW: Search service
│   │   └── docker-compose-phase2.yml     # Phase 2 services
│   │
│   ├── kubernetes/
│   │   ├── namespace.yaml                # From Phase 1
│   │   ├── configmap-phase2.yaml
│   │   ├── secrets-phase2.yaml
│   │   ├── processor-deployment.yaml     # From Phase 1
│   │   ├── embedder-deployment.yaml      # NEW: Embedding service
│   │   ├── searcher-deployment.yaml      # NEW: Search service
│   │   ├── weaviate-deployment.yaml      # NEW: Vector database
│   │   ├── elasticsearch-deployment.yaml # NEW: Search engine
│   │   └── service.yaml
│   │
│   └── helm/
│       ├── Chart-phase2.yaml
│       ├── values-phase2.yaml
│       └── templates/
│           ├── embedder-deployment.yaml
│           ├── searcher-deployment.yaml
│           ├── weaviate-deployment.yaml
│           └── elasticsearch-deployment.yaml
│
├── config/                              # Phase 2 Configuration
│   ├── init.py
│   ├── settings/
│   │   ├── init.py
│   │   ├── base.py                       # From Phase 1
│   │   ├── development.py
│   │   └── production.py
│   │
│   ├── data_sources/                     # From Phase 1
│   │   ├── init.py
│   │   ├── database_configs.py
│   │   └── api_configs.py
│   │
│   ├── ml_models/                        # NEW: ML model configurations
│   │   ├── init.py
│   │   ├── embedding_model_configs.py
│   │   ├── chunking_configs.py
│   │   ├── reranking_configs.py
│   │   └── fine_tuning_configs.py
│   │
│   ├── vector_stores/                    # NEW: Vector database configurations
│   │   ├── init.py
│   │   ├── weaviate_config.py
│   │   ├── pinecone_config.py
│   │   └── chroma_config.py
│   │
│   └── search/                           # NEW: Search configurations
│       ├── init.py
│       ├── elasticsearch_config.py
│       ├── hybrid_search_config.py
│       └── ranking_config.py
│
├── monitoring/                          # Enhanced monitoring for Phase 2
│   ├── prometheus/
│   │   ├── prometheus-phase2.yml
│   │   ├── alert_rules-phase2.yml
│   │   └── embedding_metrics.yml         # NEW: Embedding metrics
│   │
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── embedding-performance.json   # NEW: Embedding dashboard
│   │   │   ├── search-analytics.json        # NEW: Search dashboard
│   │   │   └── vector-db-metrics.json       # NEW: Vector DB dashboard
│   │   │
│   │   └── provisioning/
│   │       ├── datasources/
│   │       └── dashboards/
│   │
│   └── logging/
│       ├── filebeat-phase2.yml
│       ├── logstash-phase2.conf
│       └── search-logs-config.yml         # NEW: Search-specific logging
│
├── scripts/                             # Phase 2 utility scripts
│   ├── setup/
│   │   ├── init_database_phase2.py
│   │   ├── setup_weaviate.py             # NEW: Vector DB setup
│   │   ├── setup_elasticsearch.py        # NEW: Search engine setup
│   │   ├── setup_embedding_models.py     # NEW: Model setup
│   │   └── create_test_indices.py        # NEW: Test index creation
│   │
│   ├── data/
│   │   ├── generate_embeddings.py        # NEW: Batch embedding generation
│   │   ├── build_search_indices.py       # NEW: Index building
│   │   ├── test_search_performance.py    # NEW: Search benchmarking
│   │   └── validate_retrieval.py         # NEW: Retrieval validation
│   │
│   ├── ml/                               # NEW: ML-specific scripts
│   │   ├── fine_tune_embeddings.py       # Task 2.2.2
│   │   ├── optimize_models.py            # Task 2.2.3
│   │   ├── evaluate_embeddings.py
│   │   └── benchmark_models.py
│   │
│   └── maintenance/
│       ├── reindex_vectors.py            # NEW: Vector reindexing
│       ├── cleanup_embeddings.py         # NEW: Embedding cleanup
│       ├── backup_indices.py             # NEW: Index backup
│       └── monitor_search_quality.py     # NEW: Search quality monitoring
│
└── docs/                               # Phase 2 Documentation
├── README-phase2.md
├── SETUP-phase2.md
├── EMBEDDING-MODELS.md               # NEW: Embedding documentation
├── SEARCH-CONFIGURATION.md           # NEW: Search setup guide
├── VECTOR-DATABASES.md               # NEW: Vector DB guide
│
├── design/
│   ├── phase2-architecture.md
│   ├── embedding-pipeline.md          # NEW: Embedding pipeline design
│   ├── search-architecture.md         # NEW: Search system design
│   ├── vector-db-schema.md            # NEW: Vector database schema
│   └── retrieval-evaluation.md        # NEW: Retrieval evaluation methods
│
├── tutorials/                        # NEW: Phase 2 tutorials
│   ├── chunking-strategies.md
│   ├── embedding-fine-tuning.md
│   ├── hybrid-search-setup.md
│   └── performance-optimization.md
│
└── examples/
├── embedding-examples.md
├── search-examples.md
├── chunking-examples.md
└── reranking-examples.md