rag-system-phase3/                        # PHASE 3: Generation System and LLM Integration
├── README.md                              # Phase 3 specific documentation
├── requirements-phase3.txt                # Phase 3 dependencies (adds LLM libs, vLLM, transformers)
├── docker-compose-phase3.yml             # Phase 3 services (+ LLM inference, Redis sessions)
├── .env.phase3.example
│
├── src/
│   ├── init.py
│   │
│   ├── domain/                            # Domain Layer - Phase 3 Extensions
│   │   ├── init.py
│   │   ├── entities/
│   │   │   ├── init.py
│   │   │   ├── document.py               # From Phase 1
│   │   │   ├── chunk.py                  # From Phase 2
│   │   │   ├── embedding.py              # From Phase 2
│   │   │   ├── search_query.py           # From Phase 2
│   │   │   ├── search_result.py          # From Phase 2
│   │   │   ├── generation_request.py     # NEW: Generation request entity
│   │   │   ├── generation_response.py    # NEW: Generated response entity
│   │   │   ├── conversation.py           # NEW: Conversation entity
│   │   │   ├── session.py                # NEW: Chat session entity
│   │   │   ├── prompt_template.py        # NEW: Prompt template entity
│   │   │   └── context_window.py         # NEW: Context management entity
│   │   │
│   │   ├── value_objects/
│   │   │   ├── init.py
│   │   │   ├── document_metadata.py      # From Phase 1
│   │   │   ├── chunk_metadata.py         # From Phase 2
│   │   │   ├── embedding_vector.py       # From Phase 2
│   │   │   ├── search_filters.py         # From Phase 2
│   │   │   ├── generation_context.py     # NEW: Generation context VO
│   │   │   ├── prompt_variables.py       # NEW: Prompt variables VO
│   │   │   ├── model_parameters.py       # NEW: LLM parameters VO
│   │   │   ├── conversation_turn.py      # NEW: Single conversation turn VO
│   │   │   └── response_quality.py       # NEW: Response quality metrics VO
│   │   │
│   │   ├── repositories/                 # Repository Interfaces for Phase 3
│   │   │   ├── init.py
│   │   │   ├── document_repository.py    # From Phase 1
│   │   │   ├── chunk_repository.py       # From Phase 2
│   │   │   ├── embedding_repository.py   # From Phase 2
│   │   │   ├── search_repository.py      # From Phase 2
│   │   │   ├── conversation_repository.py # NEW: Conversation storage interface
│   │   │   ├── session_repository.py     # NEW: Session storage interface
│   │   │   ├── template_repository.py    # NEW: Template storage interface
│   │   │   └── generation_repository.py  # NEW: Generation history interface
│   │   │
│   │   ├── services/                     # Domain Services for Phase 3
│   │   │   ├── init.py
│   │   │   ├── document_processor.py     # From Phase 1
│   │   │   ├── chunking_service.py       # From Phase 2
│   │   │   ├── embedding_service.py      # From Phase 2
│   │   │   ├── retrieval_service.py      # From Phase 2
│   │   │   ├── generation_service.py     # NEW: Response generation logic
│   │   │   ├── context_service.py        # NEW: Context window management
│   │   │   ├── session_service.py        # NEW: Session management logic
│   │   │   ├── prompt_service.py         # NEW: Prompt management logic
│   │   │   └── conversation_service.py   # NEW: Conversation management logic
│   │   │
│   │   └── exceptions/
│   │       ├── init.py
│   │       ├── document_exceptions.py    # From Phase 1
│   │       ├── chunking_exceptions.py    # From Phase 2
│   │       ├── embedding_exceptions.py   # From Phase 2
│   │       ├── retrieval_exceptions.py   # From Phase 2
│   │       ├── generation_exceptions.py  # NEW: Generation errors
│   │       ├── context_exceptions.py     # NEW: Context management errors
│   │       ├── session_exceptions.py     # NEW: Session errors
│   │       └── prompt_exceptions.py      # NEW: Prompt template errors
│   │
│   ├── application/                       # Application Layer - Phase 3 Use Cases
│   │   ├── init.py
│   │   ├── use_cases/
│   │   │   ├── init.py
│   │   │   │
│   │   │   ├── data_ingestion/           # From Phase 1 (minimal changes)
│   │   │   │   ├── init.py
│   │   │   │   ├── ingest_documents.py
│   │   │   │   └── process_documents.py
│   │   │   │
│   │   │   ├── indexing/                 # From Phase 2 (minimal changes)
│   │   │   │   ├── init.py
│   │   │   │   ├── chunk_documents.py
│   │   │   │   ├── generate_embeddings.py
│   │   │   │   └── build_index.py
│   │   │   │
│   │   │   ├── retrieval/                # From Phase 2 (minimal changes)
│   │   │   │   ├── init.py
│   │   │   │   ├── semantic_search.py
│   │   │   │   ├── keyword_search.py
│   │   │   │   ├── hybrid_search.py
│   │   │   │   └── rerank_results.py
│   │   │   │
│   │   │   ├── llm_infrastructure/       # Epic 3.1: LLM Infrastructure
│   │   │   │   ├── init.py
│   │   │   │   ├── setup_inference_server.py    # Task 3.1.1
│   │   │   │   ├── fine_tune_llm.py             # Task 3.1.2
│   │   │   │   └── manage_context_window.py     # Task 3.1.3
│   │   │   │
│   │   │   ├── generation/               # Epic 3.2: Generation Pipeline
│   │   │   │   ├── init.py
│   │   │   │   ├── expand_query.py              # Task 3.2.1
│   │   │   │   ├── manage_prompt_templates.py   # Task 3.2.2
│   │   │   │   ├── generate_response.py         # Task 3.2.3
│   │   │   │   └── manage_chat_session.py       # Task 3.2.4
│   │   │   │
│   │   │   ├── conversation/             # NEW: Conversation Management
│   │   │   │   ├── init.py
│   │   │   │   ├── start_conversation.py
│   │   │   │   ├── continue_conversation.py
│   │   │   │   ├── end_conversation.py
│   │   │   │   └── summarize_conversation.py
│   │   │   │
│   │   │   └── prompt_engineering/       # NEW: Prompt Engineering
│   │   │       ├── init.py
│   │   │       ├── create_prompt_template.py
│   │   │       ├── optimize_prompts.py
│   │   │       ├── validate_prompts.py
│   │   │       └── version_prompts.py
│   │   │
│   │   ├── dto/                          # Data Transfer Objects
│   │   │   ├── init.py
│   │   │   ├── document_dto.py           # From Phase 1
│   │   │   ├── chunk_dto.py              # From Phase 2
│   │   │   ├── embedding_dto.py          # From Phase 2
│   │   │   ├── search_dto.py             # From Phase 2
│   │   │   ├── generation_dto.py         # NEW: Generation request/response
│   │   │   ├── conversation_dto.py       # NEW: Conversation data transfer
│   │   │   ├── session_dto.py            # NEW: Session data transfer
│   │   │   └── prompt_dto.py             # NEW: Prompt template transfer
│   │   │
│   │   └── interfaces/                   # Application Interfaces
│   │       ├── init.py
│   │       ├── document_processor_interface.py  # From Phase 1
│   │       ├── chunking_interface.py             # From Phase 2
│   │       ├── embedding_generator_interface.py # From Phase 2
│   │       ├── vector_store_interface.py         # From Phase 2
│   │       ├── search_engine_interface.py        # From Phase 2
│   │       ├── llm_interface.py                  # NEW: LLM interface
│   │       ├── generation_interface.py           # NEW: Generation interface
│   │       ├── conversation_interface.py         # NEW: Conversation interface
│   │       ├── session_interface.py              # NEW: Session interface
│   │       └── prompt_interface.py               # NEW: Prompt interface
│   │
│   ├── infrastructure/                    # Infrastructure Layer - Phase 3
│   │   ├── init.py
│   │   ├── persistence/
│   │   │   ├── init.py
│   │   │   ├── repositories/             # Extended repository implementations
│   │   │   │   ├── init.py
│   │   │   │   ├── postgres_document_repository.py    # From Phase 1
│   │   │   │   ├── postgres_chunk_repository.py       # From Phase 2
│   │   │   │   ├── weaviate_embedding_repository.py   # From Phase 2
│   │   │   │   ├── elasticsearch_search_repository.py # From Phase 2
│   │   │   │   ├── redis_session_repository.py        # NEW: Session storage
│   │   │   │   ├── postgres_conversation_repository.py # NEW: Conversation storage
│   │   │   │   ├── postgres_template_repository.py    # NEW: Template storage
│   │   │   │   └── postgres_generation_repository.py  # NEW: Generation history
│   │   │   │
│   │   │   ├── models/                   # Database Models
│   │   │   │   ├── init.py
│   │   │   │   ├── document_model.py     # From Phase 1
│   │   │   │   ├── chunk_model.py        # From Phase 2
│   │   │   │   ├── embedding_model.py    # From Phase 2
│   │   │   │   ├── conversation_model.py # NEW: Conversation model
│   │   │   │   ├── session_model.py      # NEW: Session model
│   │   │   │   ├── template_model.py     # NEW: Template model
│   │   │   │   └── generation_model.py   # NEW: Generation history model
│   │   │   │
│   │   │   └── migrations/
│   │   │       ├── init.py
│   │   │       ├── 001_initial_tables.py        # From Phase 1
│   │   │       ├── 002_add_metadata_fields.py   # From Phase 1
│   │   │       ├── 003_add_chunk_tables.py      # From Phase 2
│   │   │       ├── 004_add_embedding_tables.py  # From Phase 2
│   │   │       ├── 005_add_conversation_tables.py # NEW: Conversation tables
│   │   │       ├── 006_add_session_tables.py    # NEW: Session tables
│   │   │       └── 007_add_generation_tables.py # NEW: Generation tables
│   │   │
│   │   ├── external_services/
│   │   │   ├── init.py
│   │   │   ├── data_sources/             # From Phase 1 (minimal changes)
│   │   │   │   ├── init.py
│   │   │   │   ├── database_connectors/
│   │   │   │   ├── web_scrapers/
│   │   │   │   └── file_processors/
│   │   │   │
│   │   │   ├── ml_services/              # Extended ML services from Phase 2
│   │   │   │   ├── init.py
│   │   │   │   ├── embedding_models/     # From Phase 2
│   │   │   │   ├── reranking_services/    # From Phase 2
│   │   │   │   │
│   │   │   │   └── llm_services/         # NEW: Epic 3.1 & 3.2: LLM Services
│   │   │   │       ├── init.py
│   │   │   │       ├── ollama_service.py         # Task 3.1.1: Ollama LLM (llama3.2:3b)
│   │   │   │       ├── base_llm_service.py       # Base LLM service interface
│   │   │   │       ├── vllm_service.py           # Future: vLLM inference optimization
│   │   │   │       ├── huggingface_service.py    # Future: HuggingFace LLM service
│   │   │   │       ├── openai_service.py         # Future: OpenAI API service
│   │   │   │       ├── anthropic_service.py      # Future: Claude API service
│   │   │   │       └── local_llm_service.py      # Future: Local model service
│   │   │   │
│   │   │   ├── fine_tuning/              # NEW: Epic 3.1: Fine-tuning Services
│   │   │   │   ├── init.py
│   │   │   │   ├── peft_trainer.py               # Task 3.1.2: PEFT fine-tuning
│   │   │   │   ├── lora_trainer.py               # LoRA fine-tuning
│   │   │   │   ├── qlora_trainer.py              # QLoRA fine-tuning
│   │   │   │   ├── dataset_builder.py            # Training dataset creation
│   │   │   │   └── training_monitor.py           # Training monitoring
│   │   │   │
│   │   │   ├── prompt_engineering/       # NEW: Epic 3.2: Prompt Services
│   │   │   │   ├── init.py
│   │   │   │   ├── template_engine.py            # Task 3.2.2: Template management
│   │   │   │   ├── prompt_optimizer.py           # Prompt optimization
│   │   │   │   ├── few_shot_generator.py         # Few-shot example generation
│   │   │   │   ├── prompt_validator.py           # Prompt validation
│   │   │   │   └── variable_injector.py          # Variable injection
│   │   │   │
│   │   │   ├── context_management/       # NEW: Epic 3.1: Context Services
│   │   │   │   ├── init.py
│   │   │   │   ├── context_window_manager.py     # Task 3.1.3: Context management
│   │   │   │   ├── context_compressor.py         # Context compression
│   │   │   │   ├── context_summarizer.py         # Context summarization
│   │   │   │   ├── sliding_window.py             # Sliding window context
│   │   │   │   └── hierarchical_context.py       # Hierarchical context
│   │   │   │
│   │   │   ├── query_processing/         # NEW: Epic 3.2: Query Processing
│   │   │   │   ├── init.py
│   │   │   │   ├── query_expander.py             # Task 3.2.1: Query expansion
│   │   │   │   ├── intent_classifier.py          # Query intent classification
│   │   │   │   ├── entity_extractor.py           # Named entity extraction
│   │   │   │   ├── query_rewriter.py             # Query rewriting
│   │   │   │   └── synonym_generator.py          # Synonym generation
│   │   │   │
│   │   │   ├── vector_stores/            # From Phase 2 (minimal changes)
│   │   │   │   ├── init.py
│   │   │   │   ├── weaviate_client.py
│   │   │   │   ├── pinecone_client.py
│   │   │   │   └── chroma_client.py
│   │   │   │
│   │   │   └── search_engines/           # From Phase 2 (minimal changes)
│   │   │       ├── init.py
│   │   │       ├── elasticsearch_client.py
│   │   │       ├── opensearch_client.py
│   │   │       └── haystack_client.py
│   │   │
│   │   ├── orchestration/                # Extended Airflow DAGs for Phase 3
│   │   │   ├── init.py
│   │   │   ├── dags/
│   │   │   │   ├── init.py
│   │   │   │   ├── data_ingestion_dag.py         # From Phase 1
│   │   │   │   ├── chunking_dag.py               # From Phase 2
│   │   │   │   ├── embedding_generation_dag.py   # From Phase 2
│   │   │   │   ├── index_building_dag.py         # From Phase 2
│   │   │   │   ├── llm_fine_tuning_dag.py        # NEW: LLM fine-tuning pipeline
│   │   │   │   ├── model_evaluation_dag.py       # NEW: Model evaluation pipeline
│   │   │   │   └── conversation_cleanup_dag.py   # NEW: Session cleanup pipeline
│   │   │   │
│   │   │   └── operators/
│   │   │       ├── init.py
│   │   │       ├── document_processor_operator.py  # From Phase 1
│   │   │       ├── chunking_operator.py            # From Phase 2
│   │   │       ├── embedding_operator.py           # From Phase 2
│   │   │       ├── vector_store_operator.py        # From Phase 2
│   │   │       ├── llm_inference_operator.py       # NEW: LLM inference operator
│   │   │       ├── fine_tuning_operator.py         # NEW: Fine-tuning operator
│   │   │       └── generation_operator.py          # NEW: Generation operator
│   │   │
│   │   ├── storage/                      # Extended storage for Phase 3
│   │   │   ├── init.py
│   │   │   ├── object_storage/
│   │   │   │   ├── init.py
│   │   │   │   ├── minio_client.py       # From Phase 1
│   │   │   │   ├── embedding_storage.py  # From Phase 2
│   │   │   │   ├── chunk_storage.py      # From Phase 2
│   │   │   │   ├── model_storage.py      # NEW: Model artifact storage
│   │   │   │   ├── conversation_storage.py # NEW: Conversation archive storage
│   │   │   │   └── prompt_storage.py     # NEW: Prompt template storage
│   │   │   │
│   │   │   └── cache/
│   │   │       ├── init.py
│   │   │       ├── redis_client.py       # From Phase 1
│   │   │       ├── embedding_cache.py    # From Phase 2
│   │   │       ├── search_cache.py       # From Phase 2
│   │   │       ├── session_cache.py      # NEW: Session caching
│   │   │       ├── generation_cache.py   # NEW: Generation result caching
│   │   │       └── model_cache.py        # NEW: Model response caching
│   │   │
│   │   └── monitoring/                   # Enhanced monitoring for Phase 3
│   │       ├── init.py
│   │       ├── metrics/
│   │       │   ├── init.py
│   │       │   ├── embedding_metrics.py  # From Phase 2
│   │       │   ├── search_metrics.py     # From Phase 2
│   │       │   ├── generation_metrics.py # NEW: Generation performance metrics
│   │       │   ├── conversation_metrics.py # NEW: Conversation quality metrics
│   │       │   ├── llm_metrics.py        # NEW: LLM performance metrics
│   │       │   └── session_metrics.py    # NEW: Session analytics
│   │       │
│   │       └── logging/
│   │           ├── init.py
│   │           ├── search_logger.py      # From Phase 2
│   │           ├── embedding_logger.py   # From Phase 2
│   │           ├── generation_logger.py  # NEW: Generation logging
│   │           ├── conversation_logger.py # NEW: Conversation logging
│   │           └── llm_logger.py         # NEW: LLM operation logging
│   │
│   └── presentation/                     # Extended Presentation for Phase 3
│       ├── init.py
│       ├── api/                          # Extended API for generation functionality
│       │   ├── init.py
│       │   ├── v1/
│       │   │   ├── init.py
│       │   │   ├── endpoints/
│       │   │   │   ├── init.py
│       │   │   │   ├── documents.py      # From Phase 1
│       │   │   │   ├── chunks.py         # From Phase 2
│       │   │   │   ├── embeddings.py     # From Phase 2
│       │   │   │   ├── search.py         # From Phase 2
│       │   │   │   ├── chat.py           # NEW: Chat/generation endpoints
│       │   │   │   ├── conversations.py  # NEW: Conversation management
│       │   │   │   ├── sessions.py       # NEW: Session management
│       │   │   │   ├── prompts.py        # NEW: Prompt template management
│       │   │   │   ├── generation.py     # NEW: Generation endpoints
│       │   │   │   └── health.py         # Extended health check
│       │   │   │
│       │   │   └── schemas/
│       │   │       ├── init.py
│       │   │       ├── document_schemas.py  # From Phase 1
│       │   │       ├── chunk_schemas.py     # From Phase 2
│       │   │       ├── embedding_schemas.py # From Phase 2
│       │   │       ├── search_schemas.py    # From Phase 2
│       │   │       ├── chat_schemas.py      # NEW: Chat schemas
│       │   │       ├── conversation_schemas.py # NEW: Conversation schemas
│       │   │       ├── session_schemas.py   # NEW: Session schemas
│       │   │       ├── prompt_schemas.py    # NEW: Prompt schemas
│       │   │       └── generation_schemas.py # NEW: Generation schemas
│       │   │
│       │   └── middleware/
│       │       ├── init.py
│       │       ├── logging_middleware.py     # From Phase 1
│       │       ├── search_middleware.py      # From Phase 2
│       │       ├── caching_middleware.py     # From Phase 2
│       │       ├── session_middleware.py     # NEW: Session handling
│       │       ├── rate_limiting_middleware.py # NEW: Rate limiting
│       │       └── conversation_middleware.py  # NEW: Conversation context
│       │
│       ├── chatbot/                      # NEW: Chatbot Integrations
│       │   ├── init.py
│       │   ├── slack/
│       │   │   ├── init.py
│       │   │   ├── slack_bot.py          # Slack integration
│       │   │   ├── slack_handlers.py     # Slack event handlers
│       │   │   ├── slack_commands.py     # Slash commands
│       │   │   └── slack_middleware.py   # Slack-specific middleware
│       │   │
│       │   ├── teams/
│       │   │   ├── init.py
│       │   │   ├── teams_bot.py          # Microsoft Teams integration
│       │   │   ├── teams_handlers.py     # Teams event handlers
│       │   │   └── teams_cards.py        # Adaptive cards
│       │   │
│       │   ├── discord/
│       │   │   ├── init.py
│       │   │   ├── discord_bot.py        # Discord integration
│       │   │   ├── discord_handlers.py   # Discord event handlers
│       │   │   └── discord_commands.py   # Discord slash commands
│       │   │
│       │   └── web/
│       │       ├── init.py
│       │       ├── websocket_handler.py  # WebSocket chat interface
│       │       ├── chat_widget.py        # Embeddable chat widget
│       │       └── streaming_handler.py  # Streaming response handler
│       │
│       ├── web/                          # NEW: Web Dashboard for Phase 3
│       │   ├── init.py
│       │   ├── static/
│       │   │   ├── css/
│       │   │   │   ├── chat.css          # Chat interface styles
│       │   │   │   ├── dashboard.css     # Dashboard styles
│       │   │   │   └── conversation.css  # Conversation view styles
│       │   │   │
│       │   │   ├── js/
│       │   │   │   ├── chat.js           # Chat interface logic
│       │   │   │   ├── websocket.js      # WebSocket handling
│       │   │   │   ├── conversation.js   # Conversation management
│       │   │   │   └── streaming.js      # Streaming response handling
│       │   │   │
│       │   │   └── images/
│       │   │       ├── chat-icons/
│       │   │       └── ui-elements/
│       │   │
│       │   ├── templates/
│       │   │   ├── base.html
│       │   │   ├── chat.html             # NEW: Chat interface
│       │   │   ├── conversations.html    # NEW: Conversation history
│       │   │   ├── prompts.html          # NEW: Prompt management
│       │   │   ├── generation_analytics.html # NEW: Generation analytics
│       │   │   └── model_management.html # NEW: Model management
│       │   │
│       │   └── routes/
│       │       ├── init.py
│       │       ├── chat_routes.py        # NEW: Chat interface routes
│       │       ├── conversation_routes.py # NEW: Conversation routes
│       │       ├── prompt_routes.py      # NEW: Prompt management routes
│       │       ├── generation_routes.py  # NEW: Generation analytics routes
│       │       └── model_routes.py       # NEW: Model management routes
│       │
│       └── cli/                          # Extended CLI for Phase 3 operations
│           ├── init.py
│           ├── commands/
│           │   ├── init.py
│           │   ├── ingest_command.py         # From Phase 1
│           │   ├── chunk_command.py          # From Phase 2
│           │   ├── embed_command.py          # From Phase 2
│           │   ├── index_command.py          # From Phase 2
│           │   ├── search_command.py         # From Phase 2
│           │   ├── chat_command.py           # NEW: Interactive chat CLI
│           │   ├── generate_command.py       # NEW: Generation testing
│           │   ├── conversation_command.py   # NEW: Conversation management
│           │   ├── prompt_command.py         # NEW: Prompt management
│           │   ├── finetune_command.py       # NEW: Fine-tuning operations
│           │   └── benchmark_command.py      # Extended benchmarking
│           │
│           └── main.py
│
├── tests/                                # Phase 3 Tests
│   ├── init.py
│   ├── unit/
│   │   ├── init.py
│   │   ├── domain/
│   │   │   ├── test_chunk_entity.py      # From Phase 2
│   │   │   ├── test_embedding_entity.py  # From Phase 2
│   │   │   ├── test_generation_request.py # NEW: Generation entity tests
│   │   │   ├── test_conversation.py      # NEW: Conversation entity tests
│   │   │   ├── test_session.py           # NEW: Session entity tests
│   │   │   ├── test_generation_service.py # NEW: Generation service tests
│   │   │   ├── test_context_service.py   # NEW: Context service tests
│   │   │   └── test_prompt_service.py    # NEW: Prompt service tests
│   │   │
│   │   ├── application/
│   │   │   ├── test_semantic_chunking.py     # From Phase 2
│   │   │   ├── test_embedding_generation.py  # From Phase 2
│   │   │   ├── test_hybrid_search.py         # From Phase 2
│   │   │   ├── test_llm_inference.py         # NEW: LLM inference tests
│   │   │   ├── test_generation_pipeline.py   # NEW: Generation pipeline tests
│   │   │   ├── test_conversation_flow.py     # NEW: Conversation flow tests
│   │   │   ├── test_session_management.py    # NEW: Session management tests
│   │   │   └── test_prompt_engineering.py    # NEW: Prompt engineering tests
│   │   │
│   │   └── infrastructure/
│   │       ├── test_weaviate_client.py       # From Phase 2
│   │       ├── test_elasticsearch_client.py  # From Phase 2
│   │       ├── test_llm_services.py          # NEW: LLM service tests
│   │       ├── test_fine_tuning.py           # NEW: Fine-tuning tests
│   │       ├── test_prompt_engineering.py    # NEW: Prompt engineering tests
│   │       ├── test_context_management.py    # NEW: Context management tests
│   │       └── test_session_storage.py       # NEW: Session storage tests
│   │
│   ├── integration/
│   │   ├── init.py
│   │   ├── test_chunking_pipeline.py         # From Phase 2
│   │   ├── test_embedding_pipeline.py        # From Phase 2
│   │   ├── test_search_pipeline.py           # From Phase 2
│   │   ├── test_generation_pipeline.py       # NEW: End-to-end generation
│   │   ├── test_conversation_pipeline.py     # NEW: Conversation flow
│   │   ├── test_rag_pipeline.py              # NEW: Complete RAG pipeline
│   │   └── test_chatbot_integration.py       # NEW: Chatbot integration
│   │
│   ├── performance/                          # Enhanced performance tests
│   │   ├── init.py
│   │   ├── test_embedding_speed.py           # From Phase 2
│   │   ├── test_search_latency.py            # From Phase 2
│   │   ├── test_generation_latency.py        # NEW: Generation speed tests
│   │   ├── test_conversation_throughput.py   # NEW: Conversation throughput
│   │   ├── test_memory_usage.py              # Extended memory tests
│   │   ├── test_concurrent_users.py          # NEW: Concurrent user tests
│   │   └── test_model_inference_speed.py     # NEW: Model inference benchmarks
│   │
│   ├── fixtures/
│   │   ├── sample_documents/                 # From Phase 1
│   │   ├── sample_chunks/                    # From Phase 2
│   │   ├── sample_embeddings/                # From Phase 2
│   │   ├── sample_queries/                   # From Phase 2
│   │   ├── sample_conversations/             # NEW: Test conversations
│   │   ├── sample_prompts/                   # NEW: Test prompt templates
│   │   ├── sample_responses/                 # NEW: Test generated responses
│   │   └── benchmark_data/                   # Extended benchmark data
│   │
│   └── conftest.py
│
├── deployment/                           # Phase 3 Deployment
│   ├── docker/
│   │   ├── Dockerfile.processor          # From Phase 1
│   │   ├── Dockerfile.embedder           # From Phase 2
│   │   ├── Dockerfile.searcher           # From Phase 2
│   │   ├── Dockerfile.llm                # NEW: LLM inference service
│   │   ├── Dockerfile.generator          # NEW: Generation service
│   │   ├── Dockerfile.chatbot            # NEW: Chatbot service
│   │   └── docker-compose-phase3.yml     # Phase 3 services
│   │
│   ├── kubernetes/
│   │   ├── namespace.yaml                # From Phase 1
│   │   ├── configmap-phase3.yaml
│   │   ├── secrets-phase3.yaml
│   │   ├── processor-deployment.yaml     # From Phase 1
│   │   ├── embedder-deployment.yaml      # From Phase 2
│   │   ├── searcher-deployment.yaml      # From Phase 2
│   │   ├── weaviate-deployment.yaml      # From Phase 2
│   │   ├── elasticsearch-deployment.yaml # From Phase 2
│   │   ├── llm-deployment.yaml           # NEW: LLM inference deployment
│   │   ├── generator-deployment.yaml     # NEW: Generation service deployment
│   │   ├── chatbot-deployment.yaml       # NEW: Chatbot deployment
│   │   ├── redis-deployment.yaml         # NEW: Redis for sessions
│   │   └── service.yaml
│   │
│   └── helm/
│       ├── Chart-phase3.yaml
│       ├── values-phase3.yaml
│       └── templates/
│           ├── llm-deployment.yaml
│           ├── generator-deployment.yaml
│           ├── chatbot-deployment.yaml
│           └── redis-deployment.yaml
│
├── config/                              # Phase 3 Configuration
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
│   ├── ml_models/                        # Extended ML configurations
│   │   ├── init.py
│   │   ├── embedding_model_configs.py    # From Phase 2
│   │   ├── chunking_configs.py           # From Phase 2
│   │   ├── reranking_configs.py          # From Phase 2
│   │   ├── llm_configs.py                # NEW: LLM configurations
│   │   ├── fine_tuning_configs.py        # NEW: Fine-tuning configurations
│   │   └── generation_configs.py         # NEW: Generation configurations
│   │
│   ├── vector_stores/                    # From Phase 2
│   │   ├── init.py
│   │   ├── weaviate_config.py
│   │   └── chroma_config.py
│   │
│   ├── search/                           # From Phase 2
│   │   ├── init.py
│   │   ├── elasticsearch_config.py
│   │   └── hybrid_search_config.py
│   │
│   ├── chat/                             # NEW: Chat configurations
│   │   ├── init.py
│   │   ├── chatbot_configs.py
│   │   ├── session_configs.py
│   │   ├── conversation_configs.py
│   │   └── streaming_configs.py
│   │
│   └── prompts/                          # NEW: Prompt configurations
│       ├── init.py
│       ├── template_configs.py
│       ├── prompt_library.py
│       └── few_shot_examples.py
│
├── monitoring/                          # Enhanced monitoring for Phase 3
│   ├── prometheus/
│   │   ├── prometheus-phase3.yml
│   │   ├── alert_rules-phase3.yml
│   │   ├── llm_metrics.yml               # NEW: LLM metrics
│   │   └── generation_metrics.yml        # NEW: Generation metrics
│   │
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── embedding-performance.json   # From Phase 2
│   │   │   ├── search-analytics.json        # From Phase 2
│   │   │   ├── llm-performance.json         # NEW: LLM dashboard
│   │   │   ├── generation-analytics.json    # NEW: Generation dashboard
│   │   │   ├── conversation-metrics.json    # NEW: Conversation dashboard
│   │   │   └── chatbot-analytics.json       # NEW: Chatbot dashboard
│   │   │
│   │   └── provisioning/
│   │       ├── datasources/
│   │       └── dashboards/
│   │
│   └── logging/
│       ├── filebeat-phase3.yml
│       ├── logstash-phase3.conf
│       ├── llm-logs-config.yml           # NEW: LLM-specific logging
│       └── generation-logs-config.yml    # NEW: Generation logging
│
├── scripts/                             # Phase 3 utility scripts
│   ├── setup/
│   │   ├── init_database_phase3.py
│   │   ├── setup_llm_inference.py        # NEW: LLM setup
│   │   ├── setup_redis_sessions.py       # NEW: Session storage setup
│   │   ├── setup_chatbot.py              # NEW: Chatbot setup
│   │   └── create_prompt_templates.py    # NEW: Prompt template setup
│   │
│   ├── data/
│   │   ├── generate_embeddings.py        # From Phase 2
│   │   ├── test_generation.py            # NEW: Test generation pipeline
│   │   ├── benchmark_llm.py              # NEW: LLM benchmarking
│   │   └── validate_conversations.py     # NEW: Conversation validation
│   │
│   ├── ml/                               # Extended ML scripts
│   │   ├── fine_tune_embeddings.py       # From Phase 2
│   │   ├── fine_tune_llm.py              # NEW: LLM fine-tuning
│   │   ├── evaluate_generation.py        # NEW: Generation evaluation
│   │   ├── optimize_prompts.py           # NEW: Prompt optimization
│   │   └── benchmark_models.py           # Extended model benchmarking
│   │
│   ├── chat/                             # NEW: Chat-specific scripts
│   │   ├── deploy_chatbot.py
│   │   ├── test_conversation_flow.py
│   │   ├── export_conversations.py
│   │   └── analyze_chat_quality.py
│   │
│   └── maintenance/
│       ├── cleanup_sessions.py           # NEW: Session cleanup
│       ├── backup_conversations.py       # NEW: Conversation backup
│       ├── update_prompts.py             # NEW: Prompt updates
│       ├── monitor_llm_health.py         # NEW: LLM health monitoring
│       └── optimize_generation.py        # NEW: Generation optimization
│
└── docs/                               # Phase 3 Documentation
├── README-phase3.md
├── SETUP-phase3.md
├── LLM-INTEGRATION.md               # NEW: LLM integration guide
├── CHATBOT-DEPLOYMENT.md            # NEW: Chatbot deployment guide
├── CONVERSATION-MANAGEMENT.md       # NEW: Conversation guide
├── PROMPT-ENGINEERING.md            # NEW: Prompt engineering guide
│
├── design/
│   ├── phase3-architecture.md
│   ├── llm-integration.md            # NEW: LLM integration design
│   ├── generation-pipeline.md        # NEW: Generation pipeline design
│   ├── conversation-flow.md          # NEW: Conversation flow design
│   ├── session-management.md         # NEW: Session management design
│   └── chatbot-architecture.md       # NEW: Chatbot architecture
│
├── tutorials/                        # Extended tutorials
│   ├── chunking-strategies.md        # From Phase 2
│   ├── embedding-fine-tuning.md      # From Phase 2
│   ├── hybrid-search-setup.md        # From Phase 2
│   ├── llm-fine-tuning.md            # NEW: LLM fine-tuning tutorial
│   ├── prompt-engineering.md         # NEW: Prompt engineering tutorial
│   ├── chatbot-integration.md        # NEW: Chatbot integration tutorial
│   ├── conversation-design.md        # NEW: Conversation design tutorial
│   └── performance-optimization.md   # Extended optimization guide
│
└── examples/
├── embedding-examples.md         # From Phase 2
├── search-examples.md            # From Phase 2
├── generation-examples.md        # NEW: Generation examples
├── conversation-examples.md      # NEW: Conversation examples
├── prompt-examples.md            # NEW: Prompt examples
└── chatbot-examples.md           # NEW: Chatbot examples