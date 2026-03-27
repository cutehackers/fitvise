> Superseded: use `backend/botadvisor/docs/*` as the canonical source of truth.
> This file is kept only as historical migration reference.

│   │       └── alerting/                 # NEW: Advanced alerting system
│   │           ├── init.py
│   │           ├── alert_manager.py
│   │           ├── quality_alerts.py
│   │           ├── performance_alerts.py
│   │           ├── drift_alerts.py
│   │           └── notification_handler.py
│   │
│   └── presentation/                     # Extended Presentation for Phase 4
│       ├── init.py
│       ├── api/                          # Extended API for evaluation functionality
│       │   ├── init.py
│       │   ├── v1/
│       │   │   ├── init.py
│       │   │   ├── endpoints/
│       │   │   │   ├── init.py
│       │   │   │   ├── documents.py      # From Phase 1
│       │   │   │   ├── chunks.py         # From Phase 2
│       │   │   │   ├── embeddings.py     # From Phase 2
│       │   │   │   ├── search.py         # From Phase 2
│       │   │   │   ├── chat.py           # From Phase 3
│       │   │   │   ├── conversations.py  # From Phase 3
│       │   │   │   ├── sessions.py       # From Phase 3
│       │   │   │   ├── prompts.py        # From Phase 3
│       │   │   │   ├── generation.py     # From Phase 3
│       │   │   │   ├── evaluations.py    # NEW: Evaluation endpoints
│       │   │   │   ├── feedback.py       # NEW: Feedback endpoints
│       │   │   │   ├── annotations.py    # NEW: Annotation endpoints
│       │   │   │   ├── experiments.py    # NEW: Experiment endpoints
│       │   │   │   ├── benchmarks.py     # NEW: Benchmark endpoints
│       │   │   │   ├── quality.py        # NEW: Quality monitoring endpoints
│       │   │   │   └── health.py         # Extended health check
│       │   │   │
│       │   │   └── schemas/
│       │   │       ├── init.py
│       │   │       ├── document_schemas.py  # From Phase 1
│       │   │       ├── chunk_schemas.py     # From Phase 2
│       │   │       ├── embedding_schemas.py # From Phase 2
│       │   │       ├── search_schemas.py    # From Phase 2
│       │   │       ├── chat_schemas.py      # From Phase 3
│       │   │       ├── conversation_schemas.py # From Phase 3
│       │   │       ├── session_schemas.py   # From Phase 3
│       │   │       ├── prompt_schemas.py    # From Phase 3
│       │   │       ├── generation_schemas.py # From Phase 3
│       │   │       ├── evaluation_schemas.py # NEW: Evaluation schemas
│       │   │       ├── feedback_schemas.py  # NEW: Feedback schemas
│       │   │       ├── annotation_schemas.py # NEW: Annotation schemas
│       │   │       ├── experiment_schemas.py # NEW: Experiment schemas
│       │   │       ├── benchmark_schemas.py # NEW: Benchmark schemas
│       │   │       └── quality_schemas.py   # NEW: Quality schemas
│       │   │
│       │   └── middleware/
│       │       ├── init.py
│       │       ├── logging_middleware.py     # From Phase 1
│       │       ├── search_middleware.py      # From Phase 2
│       │       ├── caching_middleware.py     # From Phase 2
│       │       ├── session_middleware.py     # From Phase 3
│       │       ├── rate_limiting_middleware.py # From Phase 3
│       │       ├── conversation_middleware.py  # From Phase 3
│       │       ├── evaluation_middleware.py    # NEW: Evaluation middleware
│       │       ├── feedback_middleware.py      # NEW: Feedback middleware
│       │       └── quality_middleware.py       # NEW: Quality monitoring middleware
│       │
│       ├── chatbot/                      # Extended chatbot from Phase 3
│       │   ├── init.py
│       │   ├── slack/
│       │   │   ├── init.py
│       │   │   ├── slack_bot.py          # From Phase 3
│       │   │   ├── slack_handlers.py     # From Phase 3
│       │   │   ├── slack_commands.py     # From Phase 3
│       │   │   ├── slack_middleware.py   # From Phase 3
│       │   │   └── slack_feedback.py     # NEW: Feedback collection via Slack
│       │   │
│       │   ├── teams/
│       │   │   ├── init.py
│       │   │   ├── teams_bot.py          # From Phase 3
│       │   │   ├── teams_handlers.py     # From Phase 3
│       │   │   ├── teams_cards.py        # From Phase 3
│       │   │   └── teams_feedback.py     # NEW: Feedback collection via Teams
│       │   │
│       │   ├── discord/
│       │   │   ├── init.py
│       │   │   ├── discord_bot.py        # From Phase 3
│       │   │   ├── discord_handlers.py   # From Phase 3
│       │   │   ├── discord_commands.py   # From Phase 3
│       │   │   └── discord_feedback.py   # NEW: Feedback collection via Discord
│       │   │
│       │   └── web/
│       │       ├── init.py
│       │       ├── websocket_handler.py  # From Phase 3
│       │       ├── chat_widget.py        # From Phase 3
│       │       ├── streaming_handler.py  # From Phase 3
│       │       └── feedback_widget.py    # NEW: Feedback collection widget
│       │
│       ├── web/                          # Extended Web Dashboard for Phase 4
│       │   ├── init.py
│       │   ├── static/
│       │   │   ├── css/
│       │   │   │   ├── chat.css          # From Phase 3
│       │   │   │   ├── dashboard.css     # From Phase 3
│       │   │   │   ├── conversation.css  # From Phase 3
│       │   │   │   ├── evaluation.css    # NEW: Evaluation interface styles
│       │   │   │   ├── feedback.css      # NEW: Feedback interface styles
│       │   │   │   ├── annotation.css    # NEW: Annotation interface styles
│       │   │   │   ├── experiment.css    # NEW: Experiment interface styles
│       │   │   │   └── quality.css       # NEW: Quality monitoring styles
│       │   │   │
│       │   │   ├── js/
│       │   │   │   ├── chat.js           # From Phase 3
│       │   │   │   ├── websocket.js      # From Phase 3
│       │   │   │   ├── conversation.js   # From Phase 3
│       │   │   │   ├── streaming.js      # From Phase 3
│       │   │   │   ├── evaluation.js     # NEW: Evaluation interface logic
│       │   │   │   ├── feedback.js       # NEW: Feedback interface logic
│       │   │   │   ├── annotation.js     # NEW: Annotation interface logic
│       │   │   │   ├── experiment.js     # NEW: Experiment interface logic
│       │   │   │   ├── quality.js        # NEW: Quality monitoring logic
│       │   │   │   └── charts.js         # NEW: Advanced charting
│       │   │   │
│       │   │   └── images/
│       │   │       ├── chat-icons/       # From Phase 3
│       │   │       ├── ui-elements/      # From Phase 3
│       │   │       ├── evaluation-icons/ # NEW: Evaluation UI icons
│       │   │       └── quality-icons/    # NEW: Quality monitoring icons
│       │   │
│       │   ├── templates/
│       │   │   ├── base.html             # From Phase 3
│       │   │   ├── chat.html             # From Phase 3
│       │   │   ├── conversations.html    # From Phase 3
│       │   │   ├── prompts.html          # From Phase 3
│       │   │   ├── generation_analytics.html # From Phase 3
│       │   │   ├── model_management.html # From Phase 3
│       │   │   ├── evaluation_dashboard.html # NEW: Evaluation dashboard
│       │   │   ├── feedback_management.html  # NEW: Feedback management
│       │   │   ├── annotation_interface.html # NEW: Annotation interface
│       │   │   ├── experiment_dashboard.html # NEW: Experiment dashboard
│       │   │   ├── benchmark_results.html    # NEW: Benchmark results
│       │   │   ├── quality_monitoring.html   # NEW: Quality monitoring
│       │   │   └── analytics_overview.html   # NEW: Overall analytics
│       │   │
│       │   └── routes/
│       │       ├── init.py
│       │       ├── chat_routes.py        # From Phase 3
│       │       ├── conversation_routes.py # From Phase 3
│       │       ├── prompt_routes.py      # From Phase 3
│       │       ├── generation_routes.py  # From Phase 3
│       │       ├── model_routes.py       # From Phase 3
│       │       ├── evaluation_routes.py  # NEW: Evaluation interface routes
│       │       ├── feedback_routes.py    # NEW: Feedback management routes
│       │       ├── annotation_routes.py  # NEW: Annotation interface routes
│       │       ├── experiment_routes.py  # NEW: Experiment management routes
│       │       ├── benchmark_routes.py   # NEW: Benchmark results routes
│       │       └── quality_routes.py     # NEW: Quality monitoring routes
│       │
│       └── cli/                          # Extended CLI for Phase 4 operations
│           ├── init.py
│           ├── commands/
│           │   ├── init.py
│           │   ├── ingest_command.py         # From Phase 1
│           │   ├── chunk_command.py          # From Phase 2
│           │   ├── embed_command.py          # From Phase 2
│           │   ├── index_command.py          # From Phase 2
│           │   ├── search_command.py         # From Phase 2
│           │   ├── chat_command.py           # From Phase 3
│           │   ├── generate_command.py       # From Phase 3
│           │   ├── conversation_command.py   # From Phase 3
│           │   ├── prompt_command.py         # From Phase 3
│           │   ├── finetune_command.py       # From Phase 3
│           │   ├── evaluate_command.py       # NEW: Evaluation CLI
│           │   ├── feedback_command.py       # NEW: Feedback management CLI
│           │   ├── annotate_command.py       # NEW: Annotation CLI
│           │   ├── experiment_command.py     # NEW: Experiment management CLI
│           │   ├── benchmark_command.py      # Extended benchmarking CLI
│           │   └── quality_command.py        # NEW: Quality monitoring CLI
│           │
│           └── main.py
│
├── tests/                                # Phase 4 Tests
│   ├── init.py
│   ├── unit/
│   │   ├── init.py
│   │   ├── domain/
│   │   │   ├── test_chunk_entity.py      # From Phase 2
│   │   │   ├── test_embedding_entity.py  # From Phase 2
│   │   │   ├── test_generation_request.py # From Phase 3
│   │   │   ├── test_conversation.py      # From Phase 3
│   │   │   ├── test_session.py           # From Phase 3
│   │   │   ├── test_evaluation_result.py # NEW: Evaluation entity tests
│   │   │   ├── test_feedback.py          # NEW: Feedback entity tests
│   │   │   ├── test_annotation.py        # NEW: Annotation entity tests
│   │   │   ├── test_experiment.py        # NEW: Experiment entity tests
│   │   │   ├── test_benchmark.py         # NEW: Benchmark entity tests
│   │   │   ├── test_evaluation_service.py # NEW: Evaluation service tests
│   │   │   ├── test_feedback_service.py  # NEW: Feedback service tests
│   │   │   ├── test_annotation_service.py # NEW: Annotation service tests
│   │   │   ├── test_experiment_service.py # NEW: Experiment service tests
│   │   │   └── test_quality_service.py   # NEW: Quality service tests
│   │   │
│   │   ├── application/
│   │   │   ├── test_semantic_chunking.py     # From Phase 2
│   │   │   ├── test_embedding_generation.py  # From Phase 2
│   │   │   ├── test_hybrid_search.py         # From Phase 2
│   │   │   ├── test_llm_inference.py         # From Phase 3
│   │   │   ├── test_generation_pipeline.py   # From Phase 3
│   │   │   ├── test_conversation_flow.py     # From Phase 3
│   │   │   ├── test_evaluate_retrieval.py    # NEW: Retrieval evaluation tests
│   │   │   ├── test_evaluate_generation.py   # NEW: Generation evaluation tests
│   │   │   ├── test_detect_hallucinations.py # NEW: Hallucination detection tests
│   │   │   ├── test_collect_feedback.py      # NEW: Feedback collection tests
│   │   │   ├── test_analyze_feedback.py      # NEW: Feedback analysis tests
│   │   │   ├── test_manage_annotations.py    # NEW: Annotation management tests
│   │   │   ├── test_train_with_feedback.py   # NEW: RLHF training tests
│   │   │   ├── test_run_ab_test.py           # NEW: A/B testing tests
│   │   │   └── test_quality_assurance.py     # NEW: Quality assurance tests
│   │   │
│   │   └── infrastructure/
│   │       ├── test_weaviate_client.py       # From Phase 2
│   │       ├── test_elasticsearch_client.py  # From Phase 2
│   │       ├── test_llm_services.py          # From Phase 3
│   │       ├── test_fine_tuning.py           # From Phase 3
│   │       ├── test_ragas_evaluator.py       # NEW: Ragas evaluation tests
│   │       ├── test_llm_judge_evaluator.py   # NEW: LLM judge tests
│   │       ├── test_selfcheck_gpt.py         # NEW: SelfCheckGPT tests
│   │       ├── test_labelstudio_client.py    # NEW: LabelStudio tests
│   │       ├── test_trl_trainer.py           # NEW: TRL RLHF tests
│   │       ├── test_prometheus_client.py     # NEW: Prometheus tests
│   │       ├── test_grafana_client.py        # NEW: Grafana tests
│   │       └── test_evidently_client.py      # NEW: Evidently tests
│   │
│   ├── integration/
│   │   ├── init.py
│   │   ├── test_chunking_pipeline.py         # From Phase 2
│   │   ├── test_embedding_pipeline.py        # From Phase 2
│   │   ├── test_search_pipeline.py           # From Phase 2
│   │   ├── test_generation_pipeline.py       # From Phase 3
│   │   ├── test_conversation_pipeline.py     # From Phase 3
│   │   ├── test_rag_pipeline.py              # From Phase 3
│   │   ├── test_evaluation_pipeline.py       # NEW: End-to-end evaluation
│   │   ├── test_feedback_pipeline.py         # NEW: Feedback processing pipeline
│   │   ├── test_annotation_pipeline.py       # NEW: Annotation workflow
│   │   ├── test_experiment_pipeline.py       # NEW: Experiment management
│   │   ├── test_rlhf_pipeline.py             # NEW: RLHF training pipeline
│   │   └── test_quality_monitoring.py        # NEW: Quality monitoring pipeline
│   │
│   ├── performance/                          # Enhanced performance tests
│   │   ├── init.py
│   │   ├── test_embedding_speed.py           # From Phase 2
│   │   ├── test_search_latency.py            # From Phase 2
│   │   ├── test_generation_latency.py        # From Phase 3
│   │   ├── test_conversation_throughput.py   # From Phase 3
│   │   ├── test_evaluation_speed.py          # NEW: Evaluation performance tests
│   │   ├── test_feedback_processing_speed.py # NEW: Feedback processing tests
│   │   ├── test_annotation_throughput.py     # NEW: Annotation throughput tests
│   │   ├── test_monitoring_overhead.py       # NEW: Monitoring overhead tests
│   │   └── test_scalability.py               # NEW: System scalability tests
│   │
│   ├── evaluation/                           # NEW: Evaluation-specific tests
│   │   ├── init.py
│   │   ├── test_retrieval_metrics.py
│   │   ├── test_generation_metrics.py
│   │   ├── test_hallucination_detection.py
│   │   ├── test_quality_metrics.py
│   │   ├── test_benchmark_suite.py
│   │   └── test_evaluation_frameworks.py
│   │
│   ├── fixtures/
│   │   ├── sample_documents/                 # From Phase 1
│   │   ├── sample_chunks/                    # From Phase 2
│   │   ├── sample_embeddings/                # From Phase 2
│   │   ├── sample_queries/                   # From Phase 2
│   │   ├── sample_conversations/             # From Phase 3
│   │   ├── sample_prompts/                   # From Phase 3
│   │   ├── sample_responses/                 # From Phase 3
│   │   ├── sample_evaluations/               # NEW: Test evaluation data
│   │   ├── sample_feedback/                  # NEW: Test feedback data
│   │   ├── sample_annotations/               # NEW: Test annotation data
│   │   ├── sample_experiments/               # NEW: Test experiment data
│   │   ├── benchmark_datasets/               # NEW: Benchmark test datasets
│   │   └── ground_truth_data/                # NEW: Ground truth for evaluation
│   │
│   └── conftest.py
│
├── deployment/                           # Phase 4 Deployment
│   ├── docker/
│   │   ├── Dockerfile.processor          # From Phase 1
│   │   ├── Dockerfile.embedder           # From Phase 2
│   │   ├── Dockerfile.searcher           # From Phase 2
│   │   ├── Dockerfile.llm                # From Phase 3
│   │   ├── Dockerfile.generator          # From Phase 3
│   │   ├── Dockerfile.chatbot            # From Phase 3
│   │   ├── Dockerfile.evaluator          # NEW: Evaluation service
│   │   ├── Dockerfile.annotator          # NEW: Annotation service
│   │   ├── Dockerfile.monitor            # NEW: Monitoring service
│   │   └── docker-compose-phase4.yml     # Phase 4 services
│   │
│   ├── kubernetes/
│   │   ├── namespace.yaml                # From Phase 1
│   │   ├── configmap-phase4.yaml
│   │   ├── secrets-phase4.yaml
│   │   ├── processor-deployment.yaml     # From Phase 1
│   │   ├── embedder-deployment.yaml      # From Phase 2
│   │   ├── searcher-deployment.yaml      # From Phase 2
│   │   ├── weaviate-deployment.yaml      # From Phase 2
│   │   ├── elasticsearch-deployment.yaml # From Phase 2
│   │   ├── llm-deployment.yaml           # From Phase 3
│   │   ├── generator-deployment.yaml     # From Phase 3
│   │   ├── chatbot-deployment.yaml       # From Phase 3
│   │   ├── redis-deployment.yaml         # From Phase 3
│   │   ├── evaluator-deployment.yaml     # NEW: Evaluation service deployment
│   │   ├── labelstudio-deployment.yaml   # NEW: LabelStudio deployment
│   │   ├── prometheus-deployment.yaml    # NEW: Prometheus deployment
│   │   ├── grafana-deployment.yaml       # NEW: Grafana deployment
│   │   └── service.yaml
│   │
│   └── helm/
│       ├── Chart-phase4.yaml
│       ├── values-phase4.yaml
│       └── templates/
│           ├── evaluator-deployment.yaml
│           ├── labelstudio-deployment.yaml
│           ├── prometheus-deployment.yaml
│           └── grafana-deployment.yaml
│
├── config/                              # Phase 4 Configuration
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
│   │   ├── llm_configs.py                # From Phase 3
│   │   ├── fine_tuning_configs.py        # From Phase 3
│   │   ├── generation_configs.py         # From Phase 3
│   │   ├── evaluation_configs.py         # NEW: Evaluation configurations
│   │   └── rlhf_configs.py               # NEW: RLHF configurations
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
│   ├── chat/                             # From Phase 3
│   │   ├── init.py
│   │   ├── chatbot_configs.py
│   │   ├── session_configs.py
│   │   └── conversation_configs.py
│   │
│   ├── prompts/                          # From Phase 3
│   │   ├── init.py
│   │   ├── template_configs.py
│   │   └── prompt_library.py
│   │
│   ├── evaluation/                       # NEW: Evaluation configurations
│   │   ├── init.py
│   │   ├── metric_configs.py
│   │   ├── benchmark_configs.py
│   │   ├── evaluation_datasets.py
│   │   └── quality_thresholds.py
│   │
│   ├── feedback/                         # NEW: Feedback configurations
│   │   ├── init.py
│   │   ├── feedback_configs.py
│   │   ├── annotation_configs.py
│   │   └── labeling_guidelines.py
│   │
│   └── monitoring/                       # NEW: Monitoring configurations
│       ├── init.py
│       ├── prometheus_configs.py
│       ├── grafana_configs.py
│       ├── alert_configs.py
│       └── dashboard_configs.py
│
├── monitoring/                          # Enhanced monitoring for Phase 4
│   ├── prometheus/
│   │   ├── prometheus-phase4.yml
│   │   ├── alert_rules-phase4.yml
│   │   ├── llm_metrics.yml               # From Phase 3
│   │   ├── generation_metrics.yml        # From Phase 3
│   │   ├── evaluation_metrics.yml        # NEW: Evaluation metrics
│   │   ├── feedback_metrics.yml          # NEW: Feedback metrics
│   │   ├── quality_metrics.yml           # NEW: Quality metrics
│   │   └── experiment_metrics.yml        # NEW: Experiment metrics
│   │
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── embedding-performance.json   # From Phase 2
│   │   │   ├── search-analytics.json        # From Phase 2
│   │   │   ├── llm-performance.json         # From Phase 3
│   │   │   ├── generation-analytics.json    # From Phase 3
│   │   │   ├── conversation-metrics.json    # From Phase 3
│   │   │   ├── chatbot-analytics.json       # From Phase 3
│   │   │   ├── evaluation-dashboard.json    # NEW: Evaluation dashboard
│   │   │   ├── feedback-analytics.json      # NEW: Feedback analytics
│   │   │   ├── annotation-progress.json     # NEW: Annotation progress
│   │   │   ├── experiment-results.json      # NEW: Experiment results
│   │   │   ├── quality-monitoring.json      # NEW: Quality monitoring
│   │   │   ├── rlhf-training.json           # NEW: RLHF training dashboard
│   │   │   └── system-overview.json         # NEW: Complete system overview
│   │   │
│   │   └── provisioning/
│   │       ├── datasources/
│   │       └── dashboards/
│   │
│   ├── alerting/                         # NEW: Advanced alerting
│   │   ├── alert_rules/
│   │   │   ├── quality_alerts.yml
│   │   │   ├── performance_alerts.yml
│   │   │   ├── evaluation_alerts.yml
│   │   │   └── drift_alerts.yml
│   │   │
│   │   └── notification_channels/
│   │       ├── slack_notifications.yml
│   │       ├── email_notifications.yml
│   │       └── pagerduty_notifications.yml
│   │
│   └── logging/
│       ├── filebeat-phase4.yml
│       ├── logstash-phase4.conf
│       ├── llm-logs-config.yml           # From Phase 3
│       ├── generation-logs-config.yml    # From Phase 3
│       ├── evaluation-logs-config.yml    # NEW: Evaluation logging
│       ├── feedback-logs-config.yml      # NEW: Feedback logging
│       ├── annotation-logs-config.yml    # NEW: Annotation logging
│       └── quality-logs-config.yml       # NEW: Quality monitoring logging
│
├── scripts/                             # Phase 4 utility scripts
│   ├── setup/
│   │   ├── init_database_phase4.py
│   │   ├── setup_evaluation_services.py  # NEW: Evaluation setup
│   │   ├── setup_labelstudio.py          # NEW: LabelStudio setup
│   │   ├── setup_prometheus.py           # NEW: Prometheus setup
│   │   ├── setup_grafana.py              # NEW: Grafana setup
│   │   ├── create_evaluation_datasets.py # NEW: Evaluation dataset setup
│   │   └── setup_quality_monitoring.py   # NEW: Quality monitoring setup
│   │
│   ├── data/
│   │   ├── generate_embeddings.py        # From Phase 2
│   │   ├── test_generation.py            # From Phase 3
│   │   ├── benchmark_llm.py              # From Phase 3
│   │   ├── run_evaluation_suite.py       # NEW: Comprehensive evaluation
│   │   ├── process_feedback_data.py      # NEW: Feedback data processing
│   │   ├── export_annotations.py         # NEW: Annotation data export
│   │   ├── generate_quality_reports.py   # NEW: Quality report generation
│   │   └── validate_system_quality.py    # NEW: System quality validation
│   │
│   ├── ml/                               # Extended ML scripts
│   │   ├── fine_tune_embeddings.py       # From Phase 2
│   │   ├── fine_tune_llm.py              # From Phase 3
│   │   ├── evaluate_generation.py        # From Phase 3
│   │   ├── optimize_prompts.py           # From Phase 3
│   │   ├── benchmark_models.py           # From Phase 3
│   │   ├── train_reward_model.py         # NEW: Reward model training
│   │   ├── run_rlhf_training.py          # NEW: RLHF training
│   │   ├── evaluate_model_quality.py     # NEW: Model quality evaluation
│   │   └── compare_model_versions.py     # NEW: Model version comparison
│   │
│   ├── evaluation/                       # NEW: Evaluation-specific scripts
│   │   ├── run_retrieval_evaluation.py
│   │   ├── run_generation_evaluation.py
│   │   ├── run_hallucination_detection.py
│   │   ├── run_benchmark_suite.py
│   │   ├── generate_evaluation_report.py
│   │   └── compare_system_versions.py
│   │
│   ├── feedback/                         # NEW: Feedback-specific scripts
│   │   ├── collect_user_feedback.py
│   │   ├── analyze_feedback_trends.py
│   │   ├── export_feedback_data.py
│   │   ├── process_annotations.py
│   │   └── update_models_from_feedback.py
│   │
│   ├── monitoring/                       # NEW: Monitoring-specific scripts
│   │   ├── setup_dashboards.py
│   │   ├── configure_alerts.py
│   │   ├── test_monitoring_stack.py
│   │   ├── backup_monitoring_config.py
│   │   └── health_check_monitoring.py
│   │
│   └── maintenance/
│       ├── cleanup_sessions.py           # From Phase 3
│       ├── backup_conversations.py       # From Phase 3
│       ├── update_prompts.py             # From Phase 3
│       ├── monitor_llm_health.py         # From Phase 3
│       ├── optimize_generation.py        # From Phase 3
│       ├── cleanup_evaluation_data.py    # NEW: Evaluation data cleanup
│       ├── backup_feedback_data.py       # NEW: Feedback data backup
│       ├── rotate_logs.py                # NEW: Log rotation
│       ├── update_quality_thresholds.py  # NEW: Quality threshold updates
│       └── system_health_check.py        # NEW: Comprehensive health check
│
└── docs/                               # Phase 4 Documentation
├── README-phase4.md
├── SETUP-phase4.md
├── EVALUATION-GUIDE.md              # NEW: Evaluation setup guide
├── FEEDBACK-SYSTEM.md               # NEW: Feedback system guide
├── ANNOTATION-GUIDE.md              # NEW: Annotation guide
├── QUALITY-MONITORING.md            # NEW: Quality monitoring guide
├── RLHF-TRAINING.md                 # NEW: RLHF training guide
├── EXPERIMENTATION.md               # NEW: A/B testing guide
│
├── design/
│   ├── phase4-architecture.md
│   ├── evaluation-framework.md        # NEW: Evaluation framework design
│   ├── feedback-system-design.md     # NEW: Feedback system design
│   ├── annotation-workflow.md        # NEW: Annotation workflow design
│   ├── quality-monitoring-design.md  # NEW: Quality monitoring design
│   ├── rlhf-pipeline-design.md       # NEW: RLHF pipeline design
│   ├── experiment-design.md          # NEW: Experiment design
│   └── monitoring-architecture.md    # NEW: Monitoring architecture
│
├── evaluation/                       # NEW: Evaluation documentation
│   ├── metrics-guide.md
│   ├── benchmark-setup.md
│   ├── retrieval-evaluation.md
│   ├── generation-evaluation.md
│   ├── hallucination-detection.md
│   ├── quality-assessment.md
│   └── evaluation-best-practices.md
│
├── feedback/                         # NEW: Feedback documentation
│   ├── feedback-collection.md
│   ├── annotation-guidelines.md
│   ├── labeling-standards.md
│   ├── feedback-analysis.md
│   ├── rlhf-best-practices.md
│   └── human-in-the-loop.md
│
├── monitoring/                       # NEW: Monitoring documentation
│   ├── monitoring-setup.md
│   ├── dashboard-configuration.md
│   ├── alert-configuration.md
│   ├── metrics-collection.md
│   ├── log-management.md
│   └── troubleshooting-guide.md
│
├── tutorials/                        # Extended tutorials
│   ├── chunking-strategies.md        # From Phase 2
│   ├── embedding-fine-tuning.md      # From Phase 2
│   ├── hybrid-search-setup.md        # From Phase 2
│   ├── llm-fine-tuning.md            # From Phase 3
│   ├── prompt-engineering.md         # From Phase 3
│   ├── chatbot-integration.md        # From Phase 3
│   ├── conversation-design.md        # From Phase 3
│   ├── performance-optimization.md   # From Phase 3
│   ├── evaluation-setup.md           # NEW: Evaluation setup tutorial
│   ├── feedback-system-setup.md      # NEW: Feedback system tutorial
│   ├── annotation-workflow.md        # NEW: Annotation workflow tutorial
│   ├── rlhf-training.md              # NEW: RLHF training tutorial
│   ├── ab-testing.md                 # NEW: A/B testing tutorial
│   ├── quality-monitoring.md         # NEW: Quality monitoring tutorial
│   └── system-optimization.md        # NEW: System optimization tutorial
│
├── api/                              # NEW: API documentation
│   ├── evaluation-api.md
│   ├── feedback-api.md
│   ├── annotation-api.md
│   ├── experiment-api.md
│   ├── benchmark-api.md
│   └── quality-api.md
│
└── examples/
├── embedding-examples.md         # From Phase 2
├── search-examples.md            # From Phase 2
├── generation-examples.md        # From Phase 3
├── conversation-examples.md      # From Phase 3
├── prompt-examples.md            # From Phase 3
├── chatbot-examples.md           # From Phase 3
├── evaluation-examples.md        # NEW: Evaluation examples
├── feedback-examples.md          # NEW: Feedback examples
├── annotation-examples.md        # NEW: Annotation examples
├── experiment-examples.md        # NEW: Experiment examples
├── rlhf-examples.md              # NEW: RLHF examples
└── monitoring-examples.md        # NEW: Monitoring examplesrag-system-phase4/                        # PHASE 4: Evaluation and Quality Assurance
├── README.md                              # Phase 4 specific documentation
├── requirements-phase4.txt                # Phase 4 dependencies (adds evaluation libs, RLHF, monitoring)
├── docker-compose-phase4.yml             # Phase 4 services (+ LabelStudio, evaluation services)
├── .env.phase4.example
│
├── src/
│   ├── init.py
│   │
│   ├── domain/                            # Domain Layer - Phase 4 Extensions
│   │   ├── init.py
│   │   ├── entities/
│   │   │   ├── init.py
│   │   │   ├── document.py               # From Phase 1
│   │   │   ├── chunk.py                  # From Phase 2
│   │   │   ├── embedding.py              # From Phase 2
│   │   │   ├── search_query.py           # From Phase 2
│   │   │   ├── search_result.py          # From Phase 2
│   │   │   ├── generation_request.py     # From Phase 3
│   │   │   ├── generation_response.py    # From Phase 3
│   │   │   ├── conversation.py           # From Phase 3
│   │   │   ├── session.py                # From Phase 3
│   │   │   ├── evaluation_result.py      # NEW: Evaluation result entity
│   │   │   ├── feedback.py               # NEW: User feedback entity
│   │   │   ├── annotation.py             # NEW: Human annotation entity
│   │   │   ├── experiment.py             # NEW: A/B test experiment entity
│   │   │   ├── benchmark.py              # NEW: Benchmark test entity
│   │   │   └── quality_metric.py         # NEW: Quality metric entity
│   │   │
│   │   ├── value_objects/
│   │   │   ├── init.py
│   │   │   ├── document_metadata.py      # From Phase 1
│   │   │   ├── chunk_metadata.py         # From Phase 2
│   │   │   ├── embedding_vector.py       # From Phase 2
│   │   │   ├── search_filters.py         # From Phase 2
│   │   │   ├── generation_context.py     # From Phase 3
│   │   │   ├── conversation_turn.py      # From Phase 3
│   │   │   ├── evaluation_score.py       # NEW: Evaluation score VO
│   │   │   ├── feedback_rating.py        # NEW: Feedback rating VO
│   │   │   ├── annotation_label.py       # NEW: Annotation label VO
│   │   │   ├── metric_value.py           # NEW: Metric value VO
│   │   │   ├── benchmark_score.py        # NEW: Benchmark score VO
│   │   │   └── quality_threshold.py      # NEW: Quality threshold VO
│   │   │
│   │   ├── repositories/                 # Repository Interfaces for Phase 4
│   │   │   ├── init.py
│   │   │   ├── document_repository.py    # From Phase 1
│   │   │   ├── chunk_repository.py       # From Phase 2
│   │   │   ├── embedding_repository.py   # From Phase 2
│   │   │   ├── search_repository.py      # From Phase 2
│   │   │   ├── conversation_repository.py # From Phase 3
│   │   │   ├── session_repository.py     # From Phase 3
│   │   │   ├── evaluation_repository.py  # NEW: Evaluation results storage
│   │   │   ├── feedback_repository.py    # NEW: Feedback storage interface
│   │   │   ├── annotation_repository.py  # NEW: Annotation storage interface
│   │   │   ├── experiment_repository.py  # NEW: Experiment storage interface
│   │   │   └── benchmark_repository.py   # NEW: Benchmark storage interface
│   │   │
│   │   ├── services/                     # Domain Services for Phase 4
│   │   │   ├── init.py
│   │   │   ├── document_processor.py     # From Phase 1
│   │   │   ├── chunking_service.py       # From Phase 2
│   │   │   ├── embedding_service.py      # From Phase 2
│   │   │   ├── retrieval_service.py      # From Phase 2
│   │   │   ├── generation_service.py     # From Phase 3
│   │   │   ├── context_service.py        # From Phase 3
│   │   │   ├── evaluation_service.py     # NEW: Evaluation orchestration logic
│   │   │   ├── feedback_service.py       # NEW: Feedback processing logic
│   │   │   ├── annotation_service.py     # NEW: Annotation management logic
│   │   │   ├── experiment_service.py     # NEW: A/B testing logic
│   │   │   ├── benchmark_service.py      # NEW: Benchmarking logic
│   │   │   └── quality_service.py        # NEW: Quality assessment logic
│   │   │
│   │   └── exceptions/
│   │       ├── init.py
│   │       ├── document_exceptions.py    # From Phase 1
│   │       ├── chunking_exceptions.py    # From Phase 2
│   │       ├── embedding_exceptions.py   # From Phase 2
│   │       ├── retrieval_exceptions.py   # From Phase 2
│   │       ├── generation_exceptions.py  # From Phase 3
│   │       ├── evaluation_exceptions.py  # NEW: Evaluation errors
│   │       ├── feedback_exceptions.py    # NEW: Feedback errors
│   │       ├── annotation_exceptions.py  # NEW: Annotation errors
│   │       ├── experiment_exceptions.py  # NEW: Experiment errors
│   │       └── benchmark_exceptions.py   # NEW: Benchmark errors
│   │
│   ├── application/                       # Application Layer - Phase 4 Use Cases
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
│   │   │   ├── generation/               # From Phase 3 (minimal changes)
│   │   │   │   ├── init.py
│   │   │   │   ├── expand_query.py
│   │   │   │   ├── manage_prompt_templates.py
│   │   │   │   ├── generate_response.py
│   │   │   │   └── manage_chat_session.py
│   │   │   │
│   │   │   ├── evaluation/               # Epic 4.1: Metrics and Monitoring
│   │   │   │   ├── init.py
│   │   │   │   ├── evaluate_retrieval.py         # Task 4.1.1
│   │   │   │   ├── evaluate_generation.py        # Task 4.1.2
│   │   │   │   ├── detect_hallucinations.py      # Task 4.1.3
│   │   │   │   └── monitor_performance.py        # Task 4.1.4
│   │   │   │
│   │   │   ├── feedback/                 # Epic 4.2: Human-in-the-Loop System
│   │   │   │   ├── init.py
│   │   │   │   ├── collect_feedback.py           # Task 4.2.1
│   │   │   │   ├── analyze_feedback.py           # Task 4.2.2
│   │   │   │   ├── manage_annotations.py         # Task 4.2.3
│   │   │   │   └── train_with_feedback.py        # Task 4.2.4
│   │   │   │
│   │   │   ├── experimentation/          # NEW: A/B Testing and Experiments
│   │   │   │   ├── init.py
│   │   │   │   ├── design_experiment.py
│   │   │   │   ├── run_ab_test.py
│   │   │   │   ├── analyze_results.py
│   │   │   │   └── deploy_winner.py
│   │   │   │
│   │   │   ├── benchmarking/             # NEW: Benchmarking and Performance
│   │   │   │   ├── init.py
│   │   │   │   ├── run_benchmark.py
│   │   │   │   ├── compare_models.py
│   │   │   │   ├── measure_latency.py
│   │   │   │   └── assess_quality.py
│   │   │   │
│   │   │   └── quality_assurance/        # NEW: Quality Control
│   │   │       ├── init.py
│   │   │       ├── validate_outputs.py
│   │   │       ├── check_consistency.py
│   │   │       ├── monitor_drift.py
│   │   │       └── alert_quality_issues.py
│   │   │
│   │   ├── dto/                          # Data Transfer Objects
│   │   │   ├── init.py
│   │   │   ├── document_dto.py           # From Phase 1
│   │   │   ├── chunk_dto.py              # From Phase 2
│   │   │   ├── embedding_dto.py          # From Phase 2
│   │   │   ├── search_dto.py             # From Phase 2
│   │   │   ├── generation_dto.py         # From Phase 3
│   │   │   ├── conversation_dto.py       # From Phase 3
│   │   │   ├── evaluation_dto.py         # NEW: Evaluation data transfer
│   │   │   ├── feedback_dto.py           # NEW: Feedback data transfer
│   │   │   ├── annotation_dto.py         # NEW: Annotation data transfer
│   │   │   ├── experiment_dto.py         # NEW: Experiment data transfer
│   │   │   └── benchmark_dto.py          # NEW: Benchmark data transfer
│   │   │
│   │   └── interfaces/                   # Application Interfaces
│   │       ├── init.py
│   │       ├── document_processor_interface.py  # From Phase 1
│   │       ├── chunking_interface.py             # From Phase 2
│   │       ├── embedding_generator_interface.py # From Phase 2
│   │       ├── vector_store_interface.py         # From Phase 2
│   │       ├── search_engine_interface.py        # From Phase 2
│   │       ├── llm_interface.py                  # From Phase 3
│   │       ├── generation_interface.py           # From Phase 3
│   │       ├── evaluation_interface.py           # NEW: Evaluation interface
│   │       ├── feedback_interface.py             # NEW: Feedback interface
│   │       ├── annotation_interface.py           # NEW: Annotation interface
│   │       ├── experiment_interface.py           # NEW: Experiment interface
│   │       └── benchmark_interface.py            # NEW: Benchmark interface
│   │
│   ├── infrastructure/                    # Infrastructure Layer - Phase 4
│   │   ├── init.py
│   │   ├── persistence/
│   │   │   ├── init.py
│   │   │   ├── repositories/             # Extended repository implementations
│   │   │   │   ├── init.py
│   │   │   │   ├── postgres_document_repository.py    # From Phase 1
│   │   │   │   ├── postgres_chunk_repository.py       # From Phase 2
│   │   │   │   ├── weaviate_embedding_repository.py   # From Phase 2
│   │   │   │   ├── elasticsearch_search_repository.py # From Phase 2
│   │   │   │   ├── redis_session_repository.py        # From Phase 3
│   │   │   │   ├── postgres_conversation_repository.py # From Phase 3
│   │   │   │   ├── postgres_evaluation_repository.py  # NEW: Evaluation storage
│   │   │   │   ├── postgres_feedback_repository.py    # NEW: Feedback storage
│   │   │   │   ├── postgres_annotation_repository.py  # NEW: Annotation storage
│   │   │   │   ├── postgres_experiment_repository.py  # NEW: Experiment storage
│   │   │   │   └── postgres_benchmark_repository.py   # NEW: Benchmark storage
│   │   │   │
│   │   │   ├── models/                   # Database Models
│   │   │   │   ├── init.py
│   │   │   │   ├── document_model.py     # From Phase 1
│   │   │   │   ├── chunk_model.py        # From Phase 2
│   │   │   │   ├── embedding_model.py    # From Phase 2
│   │   │   │   ├── conversation_model.py # From Phase 3
│   │   │   │   ├── session_model.py      # From Phase 3
│   │   │   │   ├── evaluation_model.py   # NEW: Evaluation model
│   │   │   │   ├── feedback_model.py     # NEW: Feedback model
│   │   │   │   ├── annotation_model.py   # NEW: Annotation model
│   │   │   │   ├── experiment_model.py   # NEW: Experiment model
│   │   │   │   └── benchmark_model.py    # NEW: Benchmark model
│   │   │   │
│   │   │   └── migrations/
│   │   │       ├── init.py
│   │   │       ├── 001_initial_tables.py        # From Phase 1
│   │   │       ├── 002_add_metadata_fields.py   # From Phase 1
│   │   │       ├── 003_add_chunk_tables.py      # From Phase 2
│   │   │       ├── 004_add_embedding_tables.py  # From Phase 2
│   │   │       ├── 005_add_conversation_tables.py # From Phase 3
│   │   │       ├── 006_add_session_tables.py    # From Phase 3
│   │   │       ├── 007_add_generation_tables.py # From Phase 3
│   │   │       ├── 008_add_evaluation_tables.py # NEW: Evaluation tables
│   │   │       ├── 009_add_feedback_tables.py   # NEW: Feedback tables
│   │   │       ├── 010_add_annotation_tables.py # NEW: Annotation tables
│   │   │       ├── 011_add_experiment_tables.py # NEW: Experiment tables
│   │   │       └── 012_add_benchmark_tables.py  # NEW: Benchmark tables
│   │   │
│   │   ├── external_services/
│   │   │   ├── init.py
│   │   │   ├── data_sources/             # From Phase 1 (minimal changes)
│   │   │   │   ├── init.py
│   │   │   │   ├── database_connectors/
│   │   │   │   ├── web_scrapers/
│   │   │   │   └── file_processors/
│   │   │   │
│   │   │   ├── ml_services/              # Extended ML services from Phase 3
│   │   │   │   ├── init.py
│   │   │   │   ├── embedding_models/     # From Phase 2
│   │   │   │   ├── chunking_services/    # From Phase 2
│   │   │   │   ├── reranking_services/   # From Phase 2
│   │   │   │   ├── llm_services/         # From Phase 3
│   │   │   │   ├── fine_tuning/          # From Phase 3
│   │   │   │   ├── prompt_engineering/   # From Phase 3
│   │   │   │   ├── context_management/   # From Phase 3
│   │   │   │   ├── query_processing/     # From Phase 3
│   │   │   │   │
│   │   │   │   └── evaluation_services/  # NEW: Epic 4.1 & 4.2: Evaluation Services
│   │   │   │       ├── init.py
│   │   │   │       ├── retrieval_evaluators/
│   │   │   │       │   ├── init.py
│   │   │   │       │   ├── ragas_evaluator.py     # Task 4.1.1: Ragas evaluation
│   │   │   │       │   ├── precision_recall_evaluator.py
│   │   │   │       │   ├── ndcg_evaluator.py
│   │   │   │       │   └── custom_retrieval_evaluator.py
│   │   │   │       │
│   │   │   │       ├── generation_evaluators/
│   │   │   │       │   ├── init.py
│   │   │   │       │   ├── llm_judge_evaluator.py  # Task 4.1.2: LLM-as-judge
│   │   │   │       │   ├── rouge_evaluator.py
│   │   │   │       │   ├── bleu_evaluator.py
│   │   │   │       │   ├── bert_score_evaluator.py
│   │   │   │       │   └── factuality_evaluator.py
│   │   │   │       │
│   │   │   │       ├── hallucination_detectors/
│   │   │   │       │   ├── init.py
│   │   │   │       │   ├── selfcheck_gpt.py       # Task 4.1.3: SelfCheckGPT
│   │   │   │       │   ├── nli_hallucination_detector.py
│   │   │   │       │   ├── consistency_checker.py
│   │   │   │       │   └── factual_consistency_detector.py
│   │   │   │       │
│   │   │   │       ├── rlhf_services/
│   │   │   │       │   ├── init.py
│   │   │   │       │   ├── trl_trainer.py         # Task 4.2.4: TRL RLHF
│   │   │   │       │   ├── reward_model_trainer.py
│   │   │   │       │   ├── ppo_trainer.py
│   │   │   │       │   └── preference_optimizer.py
│   │   │   │       │
│   │   │   │       └── benchmark_services/
│   │   │   │           ├── init.py
│   │   │   │           ├── mteb_evaluator.py
│   │   │   │           ├── beir_evaluator.py
│   │   │   │           ├── custom_benchmark.py
│   │   │   │           └── leaderboard_submitter.py
│   │   │   │
│   │   │   ├── annotation_services/      # NEW: Epic 4.2: Annotation Services
│   │   │   │   ├── init.py
│   │   │   │   ├── labelstudio/
│   │   │   │   │   ├── init.py
│   │   │   │   │   ├── labelstudio_client.py     # Task 4.2.3: LabelStudio
│   │   │   │   │   ├── project_manager.py
│   │   │   │   │   ├── task_manager.py
│   │   │   │   │   └── annotation_exporter.py
│   │   │   │   │
│   │   │   │   ├── prodigy/
│   │   │   │   │   ├── init.py
│   │   │   │   │   ├── prodigy_client.py
│   │   │   │   │   └── recipe_manager.py
│   │   │   │   │
│   │   │   │   └── custom_annotation/
│   │   │   │       ├── init.py
│   │   │   │       ├── annotation_interface.py
│   │   │   │       ├── agreement_calculator.py
│   │   │   │       └── quality_controller.py
│   │   │   │
│   │   │   ├── monitoring_services/      # NEW: Enhanced monitoring
│   │   │   │   ├── init.py
│   │   │   │   ├── prometheus/
│   │   │   │   │   ├── init.py
│   │   │   │   │   ├── prometheus_client.py      # Task 4.1.4: Prometheus
│   │   │   │   │   ├── custom_metrics.py
│   │   │   │   │   └── alert_manager.py
│   │   │   │   │
│   │   │   │   ├── grafana/
│   │   │   │   │   ├── init.py
│   │   │   │   │   ├── grafana_client.py         # Task 4.1.4: Grafana
│   │   │   │   │   ├── dashboard_manager.py
│   │   │   │   │   └── alert_notifier.py
│   │   │   │   │
│   │   │   │   ├── datadog/
│   │   │   │   │   ├── init.py
│   │   │   │   │   ├── datadog_client.py
│   │   │   │   │   └── custom_dashboards.py
│   │   │   │   │
│   │   │   │   └── evidently/
│   │   │   │       ├── init.py
│   │   │   │       ├── evidently_client.py
│   │   │   │       ├── drift_detector.py
│   │   │   │       └── report_generator.py
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
│   │   ├── orchestration/                # Extended Airflow DAGs for Phase 4
│   │   │   ├── init.py
│   │   │   ├── dags/
│   │   │   │   ├── init.py
│   │   │   │   ├── data_ingestion_dag.py         # From Phase 1
│   │   │   │   ├── chunking_dag.py               # From Phase 2
│   │   │   │   ├── embedding_generation_dag.py   # From Phase 2
│   │   │   │   ├── index_building_dag.py         # From Phase 2
│   │   │   │   ├── llm_fine_tuning_dag.py        # From Phase 3
│   │   │   │   ├── model_evaluation_dag.py       # From Phase 3
│   │   │   │   ├── evaluation_pipeline_dag.py    # NEW: Evaluation pipeline
│   │   │   │   ├── feedback_processing_dag.py    # NEW: Feedback processing
│   │   │   │   ├── annotation_workflow_dag.py    # NEW: Annotation workflow
│   │   │   │   ├── experiment_management_dag.py  # NEW: Experiment management
│   │   │   │   ├── benchmark_execution_dag.py    # NEW: Benchmark execution
│   │   │   │   └── quality_monitoring_dag.py     # NEW: Quality monitoring
│   │   │   │
│   │   │   └── operators/
│   │   │       ├── init.py
│   │   │       ├── document_processor_operator.py  # From Phase 1
│   │   │       ├── chunking_operator.py            # From Phase 2
│   │   │       ├── embedding_operator.py           # From Phase 2
│   │   │       ├── vector_store_operator.py        # From Phase 2
│   │   │       ├── llm_inference_operator.py       # From Phase 3
│   │   │       ├── generation_operator.py          # From Phase 3
│   │   │       ├── evaluation_operator.py          # NEW: Evaluation operator
│   │   │       ├── feedback_operator.py            # NEW: Feedback operator
│   │   │       ├── annotation_operator.py          # NEW: Annotation operator
│   │   │       ├── experiment_operator.py          # NEW: Experiment operator
│   │   │       └── benchmark_operator.py           # NEW: Benchmark operator
│   │   │
│   │   ├── storage/                      # Extended storage for Phase 4
│   │   │   ├── init.py
│   │   │   ├── object_storage/
│   │   │   │   ├── init.py
│   │   │   │   ├── minio_client.py       # From Phase 1
│   │   │   │   ├── embedding_storage.py  # From Phase 2
│   │   │   │   ├── chunk_storage.py      # From Phase 2
│   │   │   │   ├── model_storage.py      # From Phase 3
│   │   │   │   ├── conversation_storage.py # From Phase 3
│   │   │   │   ├── evaluation_storage.py # NEW: Evaluation results storage
│   │   │   │   ├── feedback_storage.py   # NEW: Feedback data storage
│   │   │   │   ├── annotation_storage.py # NEW: Annotation data storage
│   │   │   │   ├── experiment_storage.py # NEW: Experiment data storage
│   │   │   │   └── benchmark_storage.py  # NEW: Benchmark data storage
│   │   │   │
│   │   │   └── cache/
│   │   │       ├── init.py
│   │   │       ├── redis_client.py       # From Phase 1
│   │   │       ├── embedding_cache.py    # From Phase 2
│   │   │       ├── search_cache.py       # From Phase 2
│   │   │       ├── session_cache.py      # From Phase 3
│   │   │       ├── generation_cache.py   # From Phase 3
│   │   │       ├── evaluation_cache.py   # NEW: Evaluation result caching
│   │   │       ├── feedback_cache.py     # NEW: Feedback caching
│   │   │       └── benchmark_cache.py    # NEW: Benchmark result caching
│   │   │
│   │   └── monitoring/                   # Enhanced monitoring for Phase 4
│   │       ├── init.py
│   │       ├── metrics/
│   │       │   ├── init.py
│   │       │   ├── embedding_metrics.py  # From Phase 2
│   │       │   ├── search_metrics.py     # From Phase 2
│   │       │   ├── generation_metrics.py # From Phase 3
│   │       │   ├── conversation_metrics.py # From Phase 3
│   │       │   ├── llm_metrics.py        # From Phase 3
│   │       │   ├── evaluation_metrics.py # NEW: Evaluation performance metrics
│   │       │   ├── feedback_metrics.py   # NEW: Feedback quality metrics
│   │       │   ├── annotation_metrics.py # NEW: Annotation quality metrics
│   │       │   ├── experiment_metrics.py # NEW: Experiment tracking metrics
│   │       │   ├── benchmark_metrics.py  # NEW: Benchmark performance metrics
│   │       │   └── quality_metrics.py    # NEW: Overall quality metrics
│   │       │
│   │       ├── logging/
│   │       │   ├── init.py
│   │       │   ├── search_logger.py      # From Phase 2
│   │       │   ├── embedding_logger.py   # From Phase 2
│   │       │   ├── generation_logger.py  # From Phase 3
│   │       │   ├── conversation_logger.py # From Phase 3
│   │       │   ├── llm_logger.py         # From Phase 3
│   │       │   ├── evaluation_logger.py  # NEW: Evaluation logging
│   │       │   ├── feedback_logger.py    # NEW: Feedback logging
│   │       │   ├── annotation_logger.py  # NEW: Annotation logging
│   │       │   ├── experiment_logger.py  # NEW: Experiment logging
│   │       │   └── quality_logger.py     # NEW: Quality monitoring logging
│   │       │
│   │       └── alerting/                 # NEW: Advanced alerting system
│   │           ├── init.py
│   │           ├── alert_manager.py
│rag-system-phase4/                        # PHASE 4: Evaluation and Quality Assurance
├── README.md                              # Phase 4 specific documentation
├── requirements-phase4