"""Use case for provisioning the Airflow ingestion environment (Task 1.2.1)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

from app.infrastructure.orchestration import (
    AirflowManager,
    AirflowDagTemplate,
    AirflowSetupReport,
)


class SetupAirflowRequest:
    """Request payload used to configure the environment generation."""

    def __init__(
        self,
        base_path: Optional[str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        compose_overrides: Optional[Dict[str, Any]] = None,
        dag_id: str = "rag_id",
        schedule: str = "@daily",
        tags: Optional[List[str]] = None,
    ) -> None:
        self.base_path = Path(base_path).expanduser().resolve() if base_path else None
        self.env_overrides = env_overrides
        self.compose_overrides = compose_overrides
        self.dag_id = dag_id
        self.schedule = schedule
        self.tags = tags or ["rag", "ingestion", "phase1"]


class SetupAirflowResponse:
    """Response containing generated artefacts and diagnostics."""

    def __init__(
        self,
        success: bool,
        report: AirflowSetupReport,
        diagnostics: Dict[str, Any],
    ) -> None:
        self.success = success
        self.report = report
        self.diagnostics = diagnostics

    @property
    def env_file(self) -> Path:
        return self.report.env_file

    @property
    def docker_compose_file(self) -> Path:
        return self.report.docker_compose_file

    @property
    def dag_file(self) -> Path:
        return self.report.dag_file

    @property
    def created_directories(self) -> List[Path]:
        return self.report.created_directories

    @property
    def warnings(self) -> List[str]:
        return self.report.warnings


class SetupAirflowUseCase:
    """Creates docker-compose, env file, folders and hello-world DAG for Airflow."""

    def __init__(self, manager: Optional[AirflowManager] = None) -> None:
        self.manager = manager or AirflowManager()

    async def execute(self, request: SetupAirflowRequest) -> SetupAirflowResponse:
        if request.base_path:
            self.manager.base_path = request.base_path
            self.manager.dags_dir = self.manager.base_path / "dags"
            self.manager.logs_dir = self.manager.base_path / "logs"
            self.manager.plugins_dir = self.manager.base_path / "plugins"

        dag_template = AirflowDagTemplate(
            dag_id=request.dag_id,
            schedule=request.schedule,
            tags=request.tags,
        )
        report = self.manager.bootstrap_environment(
            env_overrides=request.env_overrides,
            compose_overrides=request.compose_overrides,
            dag_template=dag_template,
        )
        diagnostics = self.manager.validate_generated_files(report)
        success = report.success
        return SetupAirflowResponse(success=success, report=report, diagnostics=diagnostics)
