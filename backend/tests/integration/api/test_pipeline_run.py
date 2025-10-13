from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

import os

# Minimal environment to satisfy Settings() when importing FastAPI app
os.environ.setdefault("app_name", "fitvise")
os.environ.setdefault("app_version", "0.0.0-test")
os.environ.setdefault("app_description", "test")
os.environ.setdefault("environment", "local")
os.environ.setdefault("debug", "true")
os.environ.setdefault("domain", "localhost")
os.environ.setdefault("api_host", "0.0.0.0")
os.environ.setdefault("api_port", "8000")
os.environ.setdefault("llm_base_url", "http://localhost:11434")
os.environ.setdefault("llm_model", "dummy")
os.environ.setdefault("llm_timeout", "20")
os.environ.setdefault("llm_temperature", "0.1")
os.environ.setdefault("llm_max_tokens", "128")
os.environ.setdefault("api_v1_prefix", "/api/v1")
os.environ.setdefault("cors_origins", "*")
os.environ.setdefault("cors_allow_credentials", "false")
os.environ.setdefault("cors_allow_methods", "*")
os.environ.setdefault("cors_allow_headers", "*")
os.environ.setdefault("database_url", "sqlite:///./test.db")
os.environ.setdefault("database_echo", "false")
os.environ.setdefault("vector_store_type", "faiss")
os.environ.setdefault("vector_store_path", "./vs")
os.environ.setdefault("embedding_model", "all-MiniLM-L6-v2")
os.environ.setdefault("vector_dimension", "384")
os.environ.setdefault("secret_key", "secret")
os.environ.setdefault("access_token_expire_minutes", "60")
os.environ.setdefault("algorithm", "HS256")
os.environ.setdefault("max_file_size", "10485760")
os.environ.setdefault("allowed_file_types", "pdf,docx,txt,md,csv,json")
os.environ.setdefault("upload_directory", "./uploads")
os.environ.setdefault("knowledge_base_path", "./kb")
os.environ.setdefault("auto_index_on_startup", "false")
os.environ.setdefault("index_update_interval", "3600")
os.environ.setdefault("log_level", "INFO")
os.environ.setdefault("log_file", "./app.log")
os.environ.setdefault("log_rotation", "1 week")
os.environ.setdefault("log_retention", "4 weeks")

# RAG defaults
os.environ.setdefault("rag_enabled", "true")
os.environ.setdefault("rag_data_scan_paths", "")
os.environ.setdefault("rag_max_scan_depth", "3")
os.environ.setdefault("rag_min_file_count", "1")
os.environ.setdefault("rag_model_save_path", "./models")
os.environ.setdefault("rag_export_path", "./exports")
os.environ.setdefault("rag_database_connections", "[]")
os.environ.setdefault("rag_api_endpoints", "")
os.environ.setdefault("rag_include_common_apis", "true")
os.environ.setdefault("rag_api_timeout", "10")
os.environ.setdefault("rag_api_validation_enabled", "true")
os.environ.setdefault("rag_ml_model_type", "logistic_regression")
os.environ.setdefault("rag_ml_max_features", "10000")
os.environ.setdefault("rag_ml_ngram_range_min", "1")
os.environ.setdefault("rag_ml_ngram_range_max", "2")
os.environ.setdefault("rag_ml_min_confidence", "0.6")
os.environ.setdefault("rag_ml_target_accuracy", "0.85")
os.environ.setdefault("rag_ml_synthetic_data_size", "100")
os.environ.setdefault("rag_ml_auto_retrain", "false")
os.environ.setdefault("rag_ml_retrain_interval_hours", "24")
os.environ.setdefault("rag_batch_size", "100")
os.environ.setdefault("rag_processing_timeout", "300")
os.environ.setdefault("rag_max_memory_mb", "1024")
os.environ.setdefault("rag_enable_quality_validation", "true")
os.environ.setdefault("rag_quality_threshold", "0.5")

from app.main import app


def test_pipeline_run_local_storage(tmp_path: Path):
    client = TestClient(app)
    storage_dir = tmp_path / "rag_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Minimal fake PDF bytes; processor may fall back to empty text, which is ok
    fake_pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
    b64 = base64.b64encode(fake_pdf).decode("ascii")

    payload = {
        "documents": [{"file_name": "fake.pdf", "content_base64": b64}],
        "storage_provider": "local",
        "storage_base_dir": str(storage_dir),
        "bucket_processed": "rag-processed",
    }

    resp = client.post("/api/v1/rag/pipeline/run", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["processed"] >= 1
    assert data["stored_objects"], "Expected at least one stored object"
