"""Apache Airflow environment bootstrap utilities (Task 1.2.1).

This module provides high-level helpers for preparing a dockerised Airflow setup
aligned with the Phase 1 RAG ingestion blueprint.  The goal is to make it easy to
create the minimal folders, environment variables, docker-compose manifest and a
hello-world DAG that proves the scheduler can run.
"""
from __future__ import annotations

import base64
import json
import os
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List

try:  # Optional dependency – fall back to JSON when PyYAML is unavailable.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - defensive import guard
    yaml = None  # type: ignore


def _deep_merge_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge dictionaries without mutating the inputs."""
    if not override:
        return base

    merged: Dict[str, Any] = {}
    for key in set(base) | set(override):
        if key in base and key in override:
            base_val = base[key]
            override_val = override[key]
            if isinstance(base_val, dict) and isinstance(override_val, dict):
                merged[key] = _deep_merge_dict(base_val, override_val)
            else:
                merged[key] = override_val
        elif key in override:
            merged[key] = override[key]
        else:
            merged[key] = base[key]
    return merged


@dataclass
class AirflowEnvConfig:
    """Environment configuration for the Airflow docker services."""

    executor: str = "SequentialExecutor"
    load_examples: bool = False
    expose_config: bool = True
    auth_backend: str = "airflow.api.auth.backend.basic_auth"
    webserver_port: int = 8080
    fernet_key: Optional[str] = None
    dags_folder: str = "./dags"
    logs_folder: str = "./logs"
    plugins_folder: str = "./plugins"
    admin_username: str = "admin"
    admin_password: str = "admin"
    admin_first_name: str = "RAG"
    admin_last_name: str = "Admin"
    admin_email: str = "admin@example.com"

    def __post_init__(self) -> None:
        if self.fernet_key is None:
            self.fernet_key = self.generate_fernet_key()

    @staticmethod
    def generate_fernet_key() -> str:
        """Generate a random Fernet key compatible with Airflow."""
        return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")

    def to_env_vars(self, overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Convert configuration to AIRFLOW__* env variables."""
        env = {
            "AIRFLOW__CORE__EXECUTOR": self.executor,
            "AIRFLOW__CORE__LOAD_EXAMPLES": str(self.load_examples).lower(),
            "AIRFLOW__WEBSERVER__EXPOSE_CONFIG": str(self.expose_config).lower(),
            "AIRFLOW__API__AUTH_BACKENDS": self.auth_backend,
            "AIRFLOW__WEBSERVER__WEB_SERVER_PORT": str(self.webserver_port),
            "AIRFLOW__CORE__FERNET_KEY": self.fernet_key or "",
            "AIRFLOW__CORE__DAGS_FOLDER": self.dags_folder,
            "AIRFLOW__CORE__BASE_LOG_FOLDER": self.logs_folder,
            "AIRFLOW__CORE__PLUGINS_FOLDER": self.plugins_folder,
            "AIRFLOW_VAR_ADMIN_USERNAME": self.admin_username,
            "AIRFLOW_VAR_ADMIN_PASSWORD": self.admin_password,
            "AIRFLOW_VAR_ADMIN_FIRST_NAME": self.admin_first_name,
            "AIRFLOW_VAR_ADMIN_LAST_NAME": self.admin_last_name,
            "AIRFLOW_VAR_ADMIN_EMAIL": self.admin_email,
        }
        if overrides:
            env.update(overrides)
        return env


@dataclass
class AirflowDockerConfig:
    """Docker compose configuration for the Airflow stack."""

    project_name: str = "fitvise_rag"
    airflow_image: str = "apache/airflow:2.7.3"
    airflow_uid: int = 50000
    airflow_gid: int = 0
    include_postgres: bool = True
    include_redis: bool = False
    docker_compose_filename: str = "docker-compose-airflow.yml"
    volumes: Optional[List[str]] = None
    airflow_home: str = "/opt/airflow"

    def __post_init__(self) -> None:
        if self.volumes is None:
            self.volumes = [
                "./dags:/opt/airflow/dags",
                "./logs:/opt/airflow/logs",
                "./plugins:/opt/airflow/plugins",
            ]

    def to_compose_dict(self, env_config: AirflowEnvConfig) -> Dict[str, Any]:
        """Generate a docker-compose dictionary."""
        base_services: Dict[str, Any] = {
            "airflow-init": {
                "image": self.airflow_image,
                "entrypoint": ["/bin/bash"],
                "command": [
                    "-c",
                    (
                        "airflow db check || airflow db init; "
                        "airflow users create "
                        "--username \"$AIRFLOW_VAR_ADMIN_USERNAME\" "
                        "--firstname \"$AIRFLOW_VAR_ADMIN_FIRST_NAME\" "
                        "--lastname \"$AIRFLOW_VAR_ADMIN_LAST_NAME\" "
                        "--role Admin "
                        "--email \"$AIRFLOW_VAR_ADMIN_EMAIL\" "
                        "--password \"$AIRFLOW_VAR_ADMIN_PASSWORD\" || true"
                    ),
                ],
                "env_file": [".env.airflow"],
                "user": f"{self.airflow_uid}:{self.airflow_gid}",
                "volumes": self.volumes,
            },
            "airflow-scheduler": {
                "image": self.airflow_image,
                "restart": "always",
                "command": "scheduler",
                "env_file": [".env.airflow"],
                "depends_on": {"airflow-init": {"condition": "service_completed_successfully"}},
                "volumes": self.volumes,
            },
            "airflow-webserver": {
                "image": self.airflow_image,
                "restart": "always",
                "command": "webserver",
                "env_file": [".env.airflow"],
                "ports": [f"{env_config.webserver_port}:8080"],
                "depends_on": {
                    "airflow-init": {"condition": "service_completed_successfully"},
                    "airflow-scheduler": {"condition": "service_started"},
                },
                "volumes": self.volumes,
            },
        }

        if self.include_postgres:
            base_services["postgres"] = {
                "image": "postgres:15",
                "environment": {
                    "POSTGRES_USER": "airflow",
                    "POSTGRES_PASSWORD": "airflow",
                    "POSTGRES_DB": "airflow",
                },
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "ports": ["5432:5432"],
            }
            for service_name in ("airflow-init", "airflow-scheduler", "airflow-webserver"):
                service = base_services[service_name]
                env_file = service.setdefault("env_file", [".env.airflow"])
                if ".env.postgres" not in env_file:
                    env_file.append(".env.postgres")
                service.setdefault("depends_on", {}).setdefault("postgres", {"condition": "service_healthy"})
            base_services["postgres"]["healthcheck"] = {
                "test": ["CMD", "pg_isready", "-U", "airflow"],
                "interval": "10s",
                "timeout": "5s",
                "retries": 5,
            }

        if self.include_redis:
            base_services["redis"] = {
                "image": "redis:7",
                "ports": ["6379:6379"],
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval": "10s",
                    "timeout": "3s",
                    "retries": 5,
                },
            }
            scheduler_depends = base_services["airflow-scheduler"].setdefault("depends_on", {})
            scheduler_depends.setdefault("redis", {"condition": "service_healthy"})
            web_depends = base_services["airflow-webserver"].setdefault("depends_on", {})
            web_depends.setdefault("redis", {"condition": "service_healthy"})

        compose_dict = {
            "version": "3.8",
            "name": self.project_name,
            "x-airflow-common": {
                "&airflow-common": {
                    "image": self.airflow_image,
                    "env_file": [".env.airflow"],
                    "volumes": self.volumes,
                }
            },
            "services": base_services,
            "volumes": {"postgres_data": {}} if self.include_postgres else {},
        }
        return compose_dict


@dataclass
class AirflowDAGTemplate:
    """Template for generating a hello-world DAG proving the stack works."""

    dag_id: str = "rag_hello_world"
    schedule: str = "@daily"
    catchup: bool = False
    tags: List[str] = field(default_factory=lambda: ["rag", "ingestion", "phase1"])

    def render(self) -> str:
        """Render the DAG Python source."""
        catchup_literal = "True" if self.catchup else "False"
        tags_repr = ", ".join(f"'{tag}'" for tag in self.tags)
        lines = [
            '"""Hello world DAG for FitVise RAG ingestion (auto-generated)."""',
            "from __future__ import annotations",
            "",
            "from datetime import datetime",
            "",
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "",
            "",
            "def _print_hello(**context):",
            '    msg = "Hello from FitVise RAG Airflow pipeline!"',
            "    print(msg)",
            "    return msg",
            "",
            "",
            "def _log_context(**context):",
            '    print("Execution date:", context.get("ds"))',
            '    print("Dag run conf:", context.get("dag_run"))',
            "",
            "",
            "def create_dag():",
            "    default_args = {",
            '        "owner": "fitvise",',
            '        "depends_on_past": False,',
            '        "start_date": datetime.utcnow(),',
            '        "email": ["rag-alerts@example.com"],',
            '        "email_on_failure": False,',
            '        "email_on_retry": False,',
            '        "retries": 0,',
            "    }",
            "    dag = DAG(",
            f'        dag_id="{self.dag_id}",',
            "        default_args=default_args,",
            f'        schedule_interval="{self.schedule}",',
            f"        catchup={catchup_literal},",
            f"        tags=[{tags_repr}],",
            "    )",
            "",
            "    with dag:",
            "        hello_task = PythonOperator(",
            '            task_id="hello_rag",',
            "            python_callable=_print_hello,",
            "        )",
            "",
            "        context_task = PythonOperator(",
            '            task_id="log_context",',
            "            python_callable=_log_context,",
            "        )",
            "",
            "        hello_task >> context_task",
            "",
            "    return dag",
            "",
            "",
            "dag = create_dag()",
        ]
        return "\n".join(lines) + "\n"


@dataclass
class AirflowSetupReport:
    """Summary of generated artifacts for the Airflow environment."""

    env_file: Path
    docker_compose_file: Path
    dag_file: Path
    created_directories: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.env_file.exists() and self.docker_compose_file.exists() and self.dag_file.exists()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "env_file": str(self.env_file),
            "docker_compose_file": str(self.docker_compose_file),
            "dag_file": str(self.dag_file),
            "created_directories": [str(path) for path in self.created_directories],
            "warnings": self.warnings,
            "success": self.success,
        }


class AirflowEnvironmentManager:
    """High-level helper for provisioning the Airflow ingestion environment."""

    def __init__(
        self,
        base_path: Optional[Path | str] = None,
        env_config: Optional[AirflowEnvConfig] = None,
        docker_config: Optional[AirflowDockerConfig] = None,
    ) -> None:
        self.base_path = Path(base_path or Path.cwd()).resolve()
        self.env_config = env_config or AirflowEnvConfig()
        self.docker_config = docker_config or AirflowDockerConfig()
        self.dags_dir = self.base_path / "dags"
        self.logs_dir = self.base_path / "logs"
        self.plugins_dir = self.base_path / "plugins"

    # ---------------------------------------------------------------------
    # Directory & file generation helpers
    # ---------------------------------------------------------------------
    def prepare_directories(self, create_missing: bool = True) -> List[Path]:
        """Ensure dags/logs/plugins directories exist."""
        created: List[Path] = []
        for path in (self.dags_dir, self.logs_dir, self.plugins_dir):
            if not path.exists() and create_missing:
                path.mkdir(parents=True, exist_ok=True)
                created.append(path)
        return created

    def build_env_content(self, overrides: Optional[Dict[str, str]] = None) -> str:
        env_vars = self.env_config.to_env_vars(overrides)
        return "\n".join(f"{key}={value}" for key, value in sorted(env_vars.items())) + "\n"

    def write_env_file(self, output_path: Optional[Path | str] = None, overrides: Optional[Dict[str, str]] = None) -> Path:
        env_path = Path(output_path or (self.base_path / ".env.airflow"))
        env_path.write_text(self.build_env_content(overrides), encoding="utf-8")
        return env_path

    def build_docker_compose_content(self, overrides: Optional[Dict[str, Any]] = None) -> str:
        compose_dict = self.docker_config.to_compose_dict(self.env_config)
        compose = _deep_merge_dict(compose_dict, overrides)
        if yaml is not None:
            return yaml.safe_dump(compose, sort_keys=False)  # type: ignore[attr-defined]
        return json.dumps(compose, indent=2)

    def write_docker_compose(
        self,
        output_path: Optional[Path | str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Path:
        compose_path = Path(output_path or (self.base_path / self.docker_config.docker_compose_filename))
        compose_path.write_text(self.build_docker_compose_content(overrides), encoding="utf-8")
        return compose_path

    def write_hello_world_dag(
        self,
        dag_filename: Optional[str] = None,
        dag_template: Optional[AirflowDAGTemplate] = None,
    ) -> Path:
        dag_template = dag_template or AirflowDAGTemplate()
        dag_filename = dag_filename or f"{dag_template.dag_id}.py"
        dag_path = self.dags_dir / dag_filename
        if not dag_path.parent.exists():
            dag_path.parent.mkdir(parents=True, exist_ok=True)
        dag_path.write_text(dag_template.render(), encoding="utf-8")
        return dag_path

    # ------------------------------------------------------------------
    # Composite operations
    # ------------------------------------------------------------------
    def bootstrap_environment(
        self,
        output_dir: Optional[Path | str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        compose_overrides: Optional[Dict[str, Any]] = None,
        dag_template: Optional[AirflowDAGTemplate] = None,
    ) -> AirflowSetupReport:
        """Create folders and configuration artefacts for the Airflow stack."""
        if output_dir:
            self.base_path = Path(output_dir).resolve()
            self.dags_dir = self.base_path / "dags"
            self.logs_dir = self.base_path / "logs"
            self.plugins_dir = self.base_path / "plugins"

        created_dirs = self.prepare_directories(create_missing=True)
        env_file = self.write_env_file(overrides=env_overrides)
        docker_compose_file = self.write_docker_compose(overrides=compose_overrides)
        dag_file = self.write_hello_world_dag(dag_template=dag_template)

        return AirflowSetupReport(
            env_file=env_file,
            docker_compose_file=docker_compose_file,
            dag_file=dag_file,
            created_directories=created_dirs,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate_generated_files(self, report: AirflowSetupReport) -> Dict[str, Any]:
        """Provide quick diagnostics for created artefacts."""
        diagnostics = report.as_dict()
        for path_key in ("env_file", "docker_compose_file", "dag_file"):
            path = Path(diagnostics[path_key])
            diagnostics[f"{path_key}_exists"] = path.exists()
            diagnostics[f"{path_key}_size"] = path.stat().st_size if path.exists() else 0
        diagnostics["logs_directory_exists"] = self.logs_dir.exists()
        diagnostics["plugins_directory_exists"] = self.plugins_dir.exists()
        diagnostics["dags_directory_exists"] = self.dags_dir.exists()
        return diagnostics
