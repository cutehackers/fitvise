rag-system-phase1/                        # PHASE 1: Data Acquisition and Preprocessing
├── [README.md](http://readme.md/)                              # Phase 1 specific documentation
├── requirements-phase1.txt                # Phase 1 dependencies only
├── docker-compose-phase1.yml             # Phase 1 services (Airflow, MinIO, PostgreSQL)
├── .env.phase1.example
│
├── src/
│   ├── **init**.py
│   │
│   ├── domain/                            # Domain Layer - Phase 1 Entities
│   │   ├── **init**.py
│   │   ├── entities/
│   │   │   ├── **init**.py
│   │   │   ├── [document.py](http://document.py/)               # Core document entity
│   │   │   ├── data_source.py            # Data source entity
│   │   │   └── processing_job.py         # ETL job entity
│   │   │
│   │   ├── value_objects/
│   │   │   ├── **init**.py
│   │   │   ├── document_metadata.py      # Document metadata VO
│   │   │   ├── source_info.py            # Data source information VO
│   │   │   └── quality_metrics.py        # Data quality metrics VO
│   │   │
│   │   ├── repositories/                 # Repository Interfaces for Phase 1
│   │   │   ├── **init**.py
│   │   │   ├── document_repository.py    # Document storage interface
│   │   │   ├── data_source_repository.py # Data source management interface
│   │   │   └── job_repository.py         # Processing job interface
│   │   │
│   │   ├── services/                     # Domain Services for Phase 1
│   │   │   ├── **init**.py
│   │   │   ├── document_processor.py     # Document processing logic
│   │   │   ├── source_categorizer.py     # ML-based categorization
│   │   │   └── quality_validator.py      # Data quality validation
│   │   │
│   │   └── exceptions/
│   │       ├── **init**.py
│   │       ├── document_exceptions.py
│   │       ├── source_exceptions.py
│   │       └── processing_exceptions.py
│   │
│   ├── application/                       # Application Layer - Phase 1 Use Cases
│   │   ├── **init**.py
│   │   ├── use_cases/
│   │   │   ├── **init**.py
│   │   │   │
│   │   │   ├── knowledge_sources/   # Epic 1.1: Data Source ID & Cataloging
│   │   │   │   ├── **init**.py
│   │   │   │   ├── audit_data_sources.py      # Task 1.1.1
│   │   │   │   ├── document_external_apis.py  # Task 1.1.2
│   │   │   │   └── categorize_sources.py      # Task 1.1.3
│   │   │   │
│   │   │   ├── data_ingestion/           # Epic 1.2: Core Ingestion
│   │   │   │   ├── **init**.py
│   │   │   │   ├── setup_airflow.py           # Task 1.2.1
│   │   │   │   ├── integrate_tika.py          # Task 1.2.2
│   │   │   │   ├── connect_databases.py       # Task 1.2.3
│   │   │   │   └── setup_web_scraping.py      # Task 1.2.4
│   │   │   │
│   │   │   ├── document_processing/      # Epic 1.3: Advanced Processing
│   │   │   │   ├── **init**.py
│   │   │   │   ├── process_pdfs.py            # Task 1.3.1
│   │   │   │   ├── clean_text.py              # Task 1.3.2
│   │   │   │   ├── extract_metadata.py        # Task 1.3.3
│   │   │   │   └── validate_quality.py        # Task 1.3.4
│   │   │   │
│   │   │   └── storage_management/       # Epic 1.4: Storage & Data Flow
│   │   │       ├── **init**.py
│   │   │       ├── setup_object_storage.py    # Task 1.4.1
│   │   │       └── orchestrate_etl.py         # Task 1.4.2
│   │   │
│   │   ├── dto/                          # Data Transfer Objects
│   │   │   ├── **init**.py
│   │   │   ├── document_dto.py
│   │   │   ├── source_dto.py
│   │   │   └── processing_job_dto.py
│   │   │
│   │   └── interfaces/                   # Application Interfaces
│   │       ├── **init**.py
│   │       ├── document_processor_interface.py
│   │       ├── source_connector_interface.py
│   │       └── file_processor_interface.py
│   │
│   ├── infrastructure/                    # Infrastructure Layer - Phase 1
│   │   ├── **init**.py
│   │   ├── persistence/
│   │   │   ├── **init**.py
│   │   │   ├── repositories/             # Task 1.4.1: Storage implementations
│   │   │   │   ├── **init**.py
│   │   │   │   ├── postgres_document_repository.py
│   │   │   │   ├── postgres_source_repository.py
│   │   │   │   └── postgres_job_repository.py
│   │   │   │
│   │   │   ├── models/                   # Database Models
│   │   │   │   ├── **init**.py
│   │   │   │   ├── document_model.py
│   │   │   │   ├── data_source_model.py
│   │   │   │   └── processing_job_model.py
│   │   │   │
│   │   │   └── migrations/
│   │   │       ├── **init**.py
│   │   │       ├── 001_initial_tables.py
│   │   │       └── 002_add_metadata_fields.py
│   │   │
│   │   ├── external_services/
│   │   │   ├── **init**.py
│   │   │   ├── data_sources/             # Epic 1.2 & 1.3: Data Source Connectors
│   │   │   │   ├── **init**.py
│   │   │   │   ├── database_connectors/  # Task 1.2.3
│   │   │   │   │   ├── **init**.py
│   │   │   │   │   ├── postgresql_connector.py
│   │   │   │   │   ├── mongodb_connector.py
│   │   │   │   │   ├── mysql_connector.py
│   │   │   │   │   └── base_db_connector.py
│   │   │   │   │
│   │   │   │   ├── web_scrapers/         # Task 1.2.4
│   │   │   │   │   ├── **init**.py
│   │   │   │   │   ├── confluence_scraper.py
│   │   │   │   │   ├── sharepoint_scraper.py
│   │   │   │   │   ├── slack_scraper.py
│   │   │   │   │   └── base_scraper.py
│   │   │   │   │
│   │   │   │   ├── file_processors/      # Task 1.2.2, 1.3.1
│   │   │   │   │   ├── **init**.py
│   │   │   │   │   ├── tika_processor.py      # Apache Tika integration
│   │   │   │   │   ├── docling_processor.py   # Docling PDF processing
│   │   │   │   │   ├── spacy_processor.py     # spaCy text cleaning
│   │   │   │   │   └── base_processor.py
│   │   │   │   │
│   │   │   │   └── api_clients/
│   │   │   │       ├── **init**.py
│   │   │   │       ├── google_drive_client.py
│   │   │   │       ├── notion_client.py
│   │   │   │       └── github_client.py
│   │   │   │
│   │   │   └── ml_services/              # Task 1.1.3: ML categorization
│   │   │       ├── **init**.py
│   │   │       └── categorization/
│   │   │           ├── **init**.py
│   │   │           ├── sklearn_categorizer.py
│   │   │           └── keyword_extractor.py
│   │   │
│   │   ├── orchestration/                # Epic 1.4: Airflow DAGs
│   │   │   ├── **init**.py
│   │   │   ├── dags/                     # Task 1.4.2
│   │   │   │   ├── **init**.py
│   │   │   │   ├── data_ingestion_dag.py
│   │   │   │   ├── document_processing_dag.py
│   │   │   │   └── data_quality_dag.py
│   │   │   │
│   │   │   └── operators/
│   │   │       ├── **init**.py
│   │   │       ├── document_processor_operator.py
│   │   │       ├── source_connector_operator.py
│   │   │       └── quality_check_operator.py
│   │   │
│   │   ├── storage/                      # Task 1.4.1: Object Storage
│   │   │   ├── **init**.py
│   │   │   ├── object_storage/
│   │   │   │   ├── **init**.py
│   │   │   │   ├── minio_client.py       # MinIO S3-compatible storage
│   │   │   │   └── local_storage.py      # Local development storage
│   │   │   │
│   │   │   └── cache/
│   │   │       ├── **init**.py
│   │   │       └── redis_client.py       # Basic caching for Phase 1
│   │   │
│   │   └── quality/                      # Task 1.3.4: Data Quality
│   │       ├── **init**.py
│   │       ├── validators/
│   │       │   ├── **init**.py
│   │       │   ├── great_expectations_validator.py
│   │       │   ├── schema_validator.py
│   │       │   └── content_validator.py
│   │       │
│   │       └── metrics/
│   │           ├── **init**.py
│   │           ├── quality_metrics.py
│   │           └── data_profiler.py
│   │
│   └── presentation/                     # Minimal Presentation for Phase 1
│       ├── **init**.py
│       ├── api/                          # Basic API for data ingestion
│       │   ├── **init**.py
│       │   ├── v1/
│       │   │   ├── **init**.py
│       │   │   ├── endpoints/
│       │   │   │   ├── **init**.py
│       │   │   │   ├── [documents.py](http://documents.py/)      # Document upload/management
│       │   │   │   ├── [sources.py](http://sources.py/)        # Data source management
│       │   │   │   ├── [jobs.py](http://jobs.py/)           # Processing job status
│       │   │   │   └── [health.py](http://health.py/)         # Health check
│       │   │   │
│       │   │   └── schemas/
│       │   │       ├── **init**.py
│       │   │       ├── document_schemas.py
│       │   │       ├── source_schemas.py
│       │   │       └── job_schemas.py
│       │   │
│       │   └── middleware/
│       │       ├── **init**.py
│       │       ├── logging_middleware.py
│       │       └── error_middleware.py
│       │
│       └── cli/                          # CLI for Phase 1 operations
│           ├── **init**.py
│           ├── commands/
│           │   ├── **init**.py
│           │   ├── ingest_command.py     # Manual data ingestion
│           │   ├── source_command.py     # Data source management
│           │   ├── process_command.py    # Document processing
│           │   └── validate_command.py   # Data quality validation
│           │
│           └── [main.py](http://main.py/)
│
├── tests/                                # Phase 1 Tests
│   ├── **init**.py
│   ├── unit/
│   │   ├── **init**.py
│   │   ├── domain/
│   │   │   ├── test_document_entity.py
│   │   │   ├── test_source_categorizer.py
│   │   │   └── test_quality_validator.py
│   │   │
│   │   ├── application/
│   │   │   ├── test_ingest_documents.py
│   │   │   ├── test_process_documents.py
│   │   │   └── test_validate_quality.py
│   │   │
│   │   └── infrastructure/
│   │       ├── test_database_connectors.py
│   │       ├── test_file_processors.py
│   │       └── test_web_scrapers.py
│   │
│   ├── integration/
│   │   ├── **init**.py
│   │   ├── test_database_integration.py
│   │   ├── test_file_processing_pipeline.py
│   │   └── test_airflow_dags.py
│   │
│   ├── fixtures/
│   │   ├── sample_documents/
│   │   │   ├── sample.pdf
│   │   │   ├── sample.docx
│   │   │   └── sample.csv
│   │   │
│   │   ├── test_data/
│   │   │   ├── mock_database_data.json
│   │   │   └── mock_api_responses.json
│   │   │
│   │   └── configurations/
│   │       ├── test_airflow_config.py
│   │       └── test_database_config.py
│   │
│   └── [conftest.py](http://conftest.py/)
│
├── deployment/                           # Phase 1 Deployment
│   ├── docker/
│   │   ├── Dockerfile.processor         # Document processor service
│   │   ├── Dockerfile.connector         # Data connector service
│   │   └── docker-compose-phase1.yml    # Phase 1 services only
│   │
│   ├── kubernetes/                      # Basic K8s for Phase 1
│   │   ├── namespace.yaml
│   │   ├── configmap-phase1.yaml
│   │   ├── secrets-phase1.yaml
│   │   ├── processor-deployment.yaml
│   │   ├── airflow-deployment.yaml
│   │   └── minio-deployment.yaml
│   │
│   └── helm/
│       ├── Chart-phase1.yaml
│       ├── values-phase1.yaml
│       └── templates/
│           ├── processor-deployment.yaml
│           └── airflow-deployment.yaml
│
├── config/                              # Phase 1 Configuration
│   ├── **init**.py
│   ├── settings/
│   │   ├── **init**.py
│   │   ├── [base.py](http://base.py/)
│   │   ├── [development.py](http://development.py/)
│   │   └── [production.py](http://production.py/)
│   │
│   ├── data_sources/                    # Data source configurations
│   │   ├── **init**.py
│   │   ├── database_configs.py
│   │   ├── api_configs.py
│   │   └── scraper_configs.py
│   │
│   └── processing/
│       ├── **init**.py
│       ├── file_processing_config.py
│       ├── quality_rules_config.py
│       └── airflow_config.py
│
├── monitoring/                          # Basic monitoring for Phase 1
│   ├── prometheus/
│   │   ├── prometheus-phase1.yml
│   │   └── alert_rules-phase1.yml
│   │
│   └── logging/
│       ├── filebeat-phase1.yml
│       └── logstash-phase1.conf
│
├── scripts/                             # Phase 1 utility scripts
│   ├── setup/
│   │   ├── init_database_phase1.py
│   │   ├── setup_minio.py
│   │   ├── setup_airflow.py
│   │   └── create_test_data.py
│   │
│   ├── data/
│   │   ├── sample_data_ingestion.py
│   │   ├── test_connectors.py
│   │   └── validate_pipeline.py
│   │
│   └── maintenance/
│       ├── cleanup_processed_files.py
│       ├── check_data_quality.py
│       └── backup_metadata.py
│
└── docs/                               # Phase 1 Documentation
├── [README-phase1.md](http://readme-phase1.md/)
├── [SETUP-phase1.md](http://setup-phase1.md/)
├── [DATA-SOURCES.md](http://data-sources.md/)
├── [PROCESSING-PIPELINE.md](http://processing-pipeline.md/)
│
├── design/
│   ├── [phase1-architecture.md](http://phase1-architecture.md/)
│   ├── [data-flow-diagram.md](http://data-flow-diagram.md/)
│   ├── [database-schema-phase1.md](http://database-schema-phase1.md/)
│   └── [airflow-dags-design.md](http://airflow-dags-design.md/)
│
└── examples/
├── [data-source-examples.md](http://data-source-examples.md/)
├── [processing-examples.md](http://processing-examples.md/)
└── [quality-validation-examples.md](http://quality-validation-examples.md/)