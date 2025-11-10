"""Processing job domain entity for RAG system."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from enum import Enum


class JobType(str, Enum):
    """RAG processing job types - scan, process, categorize, embed, assess, rebuild."""
    DATA_SOURCE_SCAN = "data_source_scan"
    DOCUMENT_PROCESSING = "document_processing"
    BATCH_CATEGORIZATION = "batch_categorization"
    EMBEDDING_GENERATION = "embedding_generation"
    QUALITY_ASSESSMENT = "quality_assessment"
    INDEX_REBUILD = "index_rebuild"
    HEALTH_CHECK = "health_check"
    CLEANUP = "cleanup"


class JobStatus(str, Enum):
    """Job lifecycle states - pending → running → completed/failed/cancelled."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class ProcessingJob:
    """Processing job entity - manages async RAG tasks with progress, retry & dependency handling.
    
    Examples:
        >>> job = ProcessingJob(JobType.DOCUMENT_PROCESSING, "Process PDFs", 
        ...                     "Extract text from uploaded PDFs")
        >>> job.start(total_steps=3)  # Begin execution
        >>> job.update_progress(33.0, "Extracting text...")  # Update progress
        >>> job.complete({"processed": 5, "failed": 1})  # Mark complete with results
        >>> job.is_completed()
        True
    """
    
    def __init__(
        self,
        job_type: JobType,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: JobPriority = JobPriority.NORMAL,
        id: Optional[UUID] = None,
        created_at: Optional[datetime] = None,
        created_by: Optional[str] = None
    ):
        """Initialize a processing job entity."""
        self._id = id or uuid4()
        self._job_type = job_type
        self._name = name
        self._description = description
        self._parameters = parameters or {}
        self._priority = priority
        self._created_at = created_at or datetime.utcnow()
        self._created_by = created_by or "system"
        
        # Status tracking
        self._status = JobStatus.PENDING
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        self._updated_at = self._created_at
        
        # Progress tracking
        self._progress_percentage: float = 0.0
        self._current_step: Optional[str] = None
        self._total_steps: Optional[int] = None
        self._completed_steps: int = 0
        
        # Result tracking
        self._result: Optional[Dict[str, Any]] = None
        self._error_message: Optional[str] = None
        self._error_details: Optional[Dict[str, Any]] = None
        self._logs: List[str] = []
        
        # Execution details
        self._execution_time_seconds: Optional[float] = None
        self._memory_usage_mb: Optional[float] = None
        self._cpu_usage_percent: Optional[float] = None
        
        # Retry mechanism
        self._max_retries: int = 3
        self._retry_count: int = 0
        self._retry_delay_seconds: int = 60
        self._last_retry_at: Optional[datetime] = None
        
        # Dependencies
        self._depends_on: List[UUID] = []
        self._blocks: List[UUID] = []
        
        # Scheduling
        self._scheduled_at: Optional[datetime] = None
        self._expires_at: Optional[datetime] = None
        
        # Validation
        self._validate()
    
    def _validate(self) -> None:
        """Validate the processing job entity."""
        if not self._name.strip():
            raise ValueError("Job name cannot be empty")
        
        if not self._description.strip():
            raise ValueError("Job description cannot be empty")
        
        if len(self._name) > 255:
            raise ValueError("Job name cannot exceed 255 characters")
        
        if self._progress_percentage < 0 or self._progress_percentage > 100:
            raise ValueError("Progress percentage must be between 0 and 100")
        
        if self._retry_count < 0:
            raise ValueError("Retry count cannot be negative")
        
        if self._max_retries < 0:
            raise ValueError("Max retries cannot be negative")
    
    # Properties (read-only)
    @property
    def id(self) -> UUID:
        """Get the unique identifier."""
        return self._id
    
    @property
    def job_type(self) -> JobType:
        """Get the job type."""
        return self._job_type
    
    @property
    def name(self) -> str:
        """Get the job name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Get the job description."""
        return self._description
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get job parameters."""
        return self._parameters.copy()
    
    @property
    def priority(self) -> JobPriority:
        """Get job priority."""
        return self._priority
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at
    
    @property
    def created_by(self) -> str:
        """Get creator identifier."""
        return self._created_by
    
    @property
    def status(self) -> JobStatus:
        """Get job status."""
        return self._status
    
    @property
    def started_at(self) -> Optional[datetime]:
        """Get start timestamp."""
        return self._started_at
    
    @property
    def completed_at(self) -> Optional[datetime]:
        """Get completion timestamp."""
        return self._completed_at
    
    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self._updated_at
    
    @property
    def progress_percentage(self) -> float:
        """Get progress percentage."""
        return self._progress_percentage
    
    @property
    def current_step(self) -> Optional[str]:
        """Get current step description."""
        return self._current_step
    
    @property
    def total_steps(self) -> Optional[int]:
        """Get total number of steps."""
        return self._total_steps
    
    @property
    def completed_steps(self) -> int:
        """Get number of completed steps."""
        return self._completed_steps
    
    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """Get job result."""
        return self._result.copy() if self._result else None
    
    @property
    def error_message(self) -> Optional[str]:
        """Get error message."""
        return self._error_message
    
    @property
    def error_details(self) -> Optional[Dict[str, Any]]:
        """Get error details."""
        return self._error_details.copy() if self._error_details else None
    
    @property
    def logs(self) -> List[str]:
        """Get job logs."""
        return self._logs.copy()
    
    @property
    def execution_time_seconds(self) -> Optional[float]:
        """Get execution time in seconds."""
        return self._execution_time_seconds
    
    @property
    def memory_usage_mb(self) -> Optional[float]:
        """Get memory usage in MB."""
        return self._memory_usage_mb
    
    @property
    def cpu_usage_percent(self) -> Optional[float]:
        """Get CPU usage percentage."""
        return self._cpu_usage_percent
    
    @property
    def retry_count(self) -> int:
        """Get current retry count."""
        return self._retry_count
    
    @property
    def max_retries(self) -> int:
        """Get maximum retries."""
        return self._max_retries
    
    @property
    def depends_on(self) -> List[UUID]:
        """Get job dependencies."""
        return self._depends_on.copy()
    
    @property
    def blocks(self) -> List[UUID]:
        """Get jobs blocked by this job."""
        return self._blocks.copy()
    
    @property
    def scheduled_at(self) -> Optional[datetime]:
        """Get scheduled execution time."""
        return self._scheduled_at
    
    @property
    def expires_at(self) -> Optional[datetime]:
        """Get expiration time."""
        return self._expires_at
    
    # Business methods
    def update_priority(self, priority: JobPriority) -> None:
        """Update job priority."""
        if self._status not in [JobStatus.PENDING, JobStatus.QUEUED]:
            raise ValueError("Cannot change priority of running or completed job")
        
        self._priority = priority
        self._updated_at = datetime.utcnow()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update job parameters."""
        if self._status not in [JobStatus.PENDING, JobStatus.QUEUED]:
            raise ValueError("Cannot change parameters of running or completed job")
        
        self._parameters = parameters.copy()
        self._updated_at = datetime.utcnow()
    
    def schedule(self, scheduled_at: datetime, expires_at: Optional[datetime] = None) -> None:
        """Schedule the job for execution."""
        if self._status != JobStatus.PENDING:
            raise ValueError("Can only schedule pending jobs")
        
        self._scheduled_at = scheduled_at
        self._expires_at = expires_at
        self._status = JobStatus.QUEUED
        self._updated_at = datetime.utcnow()
    
    def start(self, total_steps: Optional[int] = None) -> None:
        """Start job execution."""
        if self._status not in [JobStatus.QUEUED, JobStatus.PENDING, JobStatus.RETRYING]:
            raise ValueError(f"Cannot start job in status {self._status}")
        
        self._status = JobStatus.RUNNING
        self._started_at = datetime.utcnow()
        self._total_steps = total_steps
        self._completed_steps = 0
        self._progress_percentage = 0.0
        self._current_step = "Starting..."
        self._updated_at = datetime.utcnow()
        
        self.add_log("Job execution started")
    
    def update_progress(self, percentage: float, current_step: Optional[str] = None) -> None:
        """Update job progress."""
        if self._status != JobStatus.RUNNING:
            raise ValueError("Can only update progress of running jobs")
        
        if percentage < 0 or percentage > 100:
            raise ValueError("Progress percentage must be between 0 and 100")
        
        self._progress_percentage = percentage
        if current_step:
            self._current_step = current_step
        
        self._updated_at = datetime.utcnow()
    
    def complete_step(self, step_name: Optional[str] = None) -> None:
        """Mark a step as completed."""
        if self._status != JobStatus.RUNNING:
            raise ValueError("Can only complete steps of running jobs")
        
        self._completed_steps += 1
        
        if self._total_steps:
            self._progress_percentage = (self._completed_steps / self._total_steps) * 100
        
        if step_name:
            self.add_log(f"Completed step: {step_name}")
        
        self._updated_at = datetime.utcnow()
    
    def complete(self, result: Optional[Dict[str, Any]] = None, 
                execution_time: Optional[float] = None,
                memory_usage: Optional[float] = None,
                cpu_usage: Optional[float] = None) -> None:
        """Complete the job successfully."""
        if self._status != JobStatus.RUNNING:
            raise ValueError("Can only complete running jobs")
        
        self._status = JobStatus.COMPLETED
        self._completed_at = datetime.utcnow()
        self._progress_percentage = 100.0
        self._current_step = "Completed"
        self._result = result or {}
        self._execution_time_seconds = execution_time
        self._memory_usage_mb = memory_usage
        self._cpu_usage_percent = cpu_usage
        self._updated_at = datetime.utcnow()
        
        self.add_log("Job completed successfully")
    
    def fail(self, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Mark the job as failed."""
        if self._status not in [JobStatus.RUNNING, JobStatus.RETRYING]:
            raise ValueError("Can only fail running or retrying jobs")
        
        self._error_message = error_message
        self._error_details = error_details or {}
        
        # Check if we should retry
        if self._retry_count < self._max_retries:
            self._status = JobStatus.RETRYING
            self._retry_count += 1
            self._last_retry_at = datetime.utcnow()
            self.add_log(f"Job failed, scheduling retry {self._retry_count}/{self._max_retries}: {error_message}")
        else:
            self._status = JobStatus.FAILED
            self._completed_at = datetime.utcnow()
            self.add_log(f"Job failed permanently after {self._retry_count} retries: {error_message}")
        
        self._updated_at = datetime.utcnow()
    
    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel the job."""
        if self._status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise ValueError("Cannot cancel completed, failed, or already cancelled job")
        
        self._status = JobStatus.CANCELLED
        self._completed_at = datetime.utcnow()
        self._updated_at = datetime.utcnow()
        
        log_message = f"Job cancelled"
        if reason:
            log_message += f": {reason}"
        self.add_log(log_message)
    
    def reset_for_retry(self) -> None:
        """Reset job state for retry."""
        if self._status != JobStatus.RETRYING:
            raise ValueError("Can only reset jobs in retrying status")
        
        self._status = JobStatus.QUEUED
        self._progress_percentage = 0.0
        self._current_step = None
        self._completed_steps = 0
        self._error_message = None
        self._error_details = None
        self._updated_at = datetime.utcnow()
        
        self.add_log(f"Job reset for retry attempt {self._retry_count}")
    
    def add_dependency(self, job_id: UUID) -> None:
        """Add a job dependency."""
        if job_id not in self._depends_on:
            self._depends_on.append(job_id)
            self._updated_at = datetime.utcnow()
    
    def remove_dependency(self, job_id: UUID) -> None:
        """Remove a job dependency."""
        if job_id in self._depends_on:
            self._depends_on.remove(job_id)
            self._updated_at = datetime.utcnow()
    
    def add_blocked_job(self, job_id: UUID) -> None:
        """Add a job that this job blocks."""
        if job_id not in self._blocks:
            self._blocks.append(job_id)
            self._updated_at = datetime.utcnow()
    
    def add_log(self, message: str) -> None:
        """Add a log entry."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"{timestamp}: {message}"
        self._logs.append(log_entry)
        
        # Keep only the last 100 log entries
        if len(self._logs) > 100:
            self._logs = self._logs[-100:]
        
        self._updated_at = datetime.utcnow()
    
    # Status methods
    def is_pending(self) -> bool:
        """Check if job is pending."""
        return self._status == JobStatus.PENDING
    
    def is_queued(self) -> bool:
        """Check if job is queued."""
        return self._status == JobStatus.QUEUED
    
    def is_running(self) -> bool:
        """Check if job is running."""
        return self._status == JobStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self._status == JobStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if job is failed."""
        return self._status == JobStatus.FAILED
    
    def is_cancelled(self) -> bool:
        """Check if job is cancelled."""
        return self._status == JobStatus.CANCELLED
    
    def is_retrying(self) -> bool:
        """Check if job is retrying."""
        return self._status == JobStatus.RETRYING
    
    def is_finished(self) -> bool:
        """Check if job is in a terminal state."""
        return self._status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self._retry_count < self._max_retries
    
    def is_ready_to_run(self) -> bool:
        """Check if job is ready to run (no blocking dependencies)."""
        if self._status not in [JobStatus.QUEUED, JobStatus.PENDING]:
            return False
        
        # Check if scheduled time has passed
        if self._scheduled_at and datetime.utcnow() < self._scheduled_at:
            return False
        
        # Check if expired
        if self._expires_at and datetime.utcnow() > self._expires_at:
            return False
        
        return True
    
    def is_expired(self) -> bool:
        """Check if job has expired."""
        return self._expires_at is not None and datetime.utcnow() > self._expires_at
    
    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if not self._started_at:
            return None

        end_time = self._completed_at or datetime.utcnow()
        return (end_time - self._started_at).total_seconds()

    def get_progress_dict(self) -> Dict[str, Any]:
        """Get progress tracking information as a dictionary.

        Returns:
            Dictionary containing progress, status, and timing information
        """
        return {
            "progress_percentage": self._progress_percentage,
            "current_step": self._current_step,
            "completed_steps": self._completed_steps,
            "total_steps": self._total_steps,
            "status": self._status.value,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "completed_at": self._completed_at.isoformat() if self._completed_at else None,
            "updated_at": self._updated_at.isoformat(),
            "execution_time_seconds": self.get_duration(),
            "memory_usage_mb": self._memory_usage_mb,
            "cpu_usage_percent": self._cpu_usage_percent,
            "retry_count": self._retry_count,
            "max_retries": self._max_retries,
            "has_error": self._error_message is not None,
            "error_message": self._error_message,
            "log_count": len(self._logs)
        }
    
    def get_status_summary(self) -> dict:
        """Get comprehensive status summary."""
        return {
            "id": str(self._id),
            "type": self._job_type.value,
            "name": self._name,
            "status": self._status.value,
            "priority": self._priority.value,
            "progress": self._progress_percentage,
            "current_step": self._current_step,
            "completed_steps": self._completed_steps,
            "total_steps": self._total_steps,
            "created_at": self._created_at.isoformat(),
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "completed_at": self._completed_at.isoformat() if self._completed_at else None,
            "duration_seconds": self.get_duration(),
            "retry_count": self._retry_count,
            "max_retries": self._max_retries,
            "has_error": self._error_message is not None,
            "error_message": self._error_message,
            "dependencies": len(self._depends_on),
            "blocks": len(self._blocks),
            "is_ready": self.is_ready_to_run(),
            "is_expired": self.is_expired(),
            "log_count": len(self._logs)
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"ProcessingJob(id={self._id}, name='{self._name}', status={self._status.value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"ProcessingJob(id={self._id}, type={self._job_type.value}, "
                f"name='{self._name}', status={self._status.value}, "
                f"progress={self._progress_percentage}%)")