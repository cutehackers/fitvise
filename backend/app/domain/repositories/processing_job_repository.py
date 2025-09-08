"""Processing job repository interface."""
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from app.domain.entities.processing_job import ProcessingJob, JobStatus, JobType, JobPriority


class ProcessingJobRepository(ABC):
    """Abstract repository for processing job persistence."""
    
    @abstractmethod
    async def save(self, job: ProcessingJob) -> ProcessingJob:
        """Save a processing job."""
        pass
    
    @abstractmethod
    async def find_by_id(self, job_id: UUID) -> Optional[ProcessingJob]:
        """Find a processing job by ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> List[ProcessingJob]:
        """Find processing jobs by name."""
        pass
    
    @abstractmethod
    async def find_by_type(self, job_type: JobType) -> List[ProcessingJob]:
        """Find processing jobs by type."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: JobStatus) -> List[ProcessingJob]:
        """Find processing jobs by status."""
        pass
    
    @abstractmethod
    async def find_by_priority(self, priority: JobPriority) -> List[ProcessingJob]:
        """Find processing jobs by priority."""
        pass
    
    @abstractmethod
    async def find_by_creator(self, created_by: str) -> List[ProcessingJob]:
        """Find processing jobs by creator."""
        pass
    
    @abstractmethod
    async def find_pending_jobs(self) -> List[ProcessingJob]:
        """Find pending jobs."""
        pass
    
    @abstractmethod
    async def find_queued_jobs(self) -> List[ProcessingJob]:
        """Find queued jobs."""
        pass
    
    @abstractmethod
    async def find_running_jobs(self) -> List[ProcessingJob]:
        """Find running jobs."""
        pass
    
    @abstractmethod
    async def find_completed_jobs(self) -> List[ProcessingJob]:
        """Find completed jobs."""
        pass
    
    @abstractmethod
    async def find_failed_jobs(self) -> List[ProcessingJob]:
        """Find failed jobs."""
        pass
    
    @abstractmethod
    async def find_retrying_jobs(self) -> List[ProcessingJob]:
        """Find jobs in retry state."""
        pass
    
    @abstractmethod
    async def find_ready_to_run(self) -> List[ProcessingJob]:
        """Find jobs ready to run (no blocking dependencies)."""
        pass
    
    @abstractmethod
    async def find_expired_jobs(self) -> List[ProcessingJob]:
        """Find expired jobs."""
        pass
    
    @abstractmethod
    async def find_jobs_by_dependency(self, dependency_id: UUID) -> List[ProcessingJob]:
        """Find jobs that depend on a specific job."""
        pass
    
    @abstractmethod
    async def find_blocked_jobs(self, blocking_job_id: UUID) -> List[ProcessingJob]:
        """Find jobs blocked by a specific job."""
        pass
    
    @abstractmethod
    async def find_recent_jobs(self, limit: int = 100) -> List[ProcessingJob]:
        """Find recent jobs (ordered by creation time)."""
        pass
    
    @abstractmethod
    async def delete(self, job_id: UUID) -> bool:
        """Delete a processing job."""
        pass
    
    @abstractmethod
    async def delete_completed_jobs(self, older_than_days: int) -> int:
        """Delete completed jobs older than specified days."""
        pass
    
    @abstractmethod
    async def count_all(self) -> int:
        """Count total number of jobs."""
        pass
    
    @abstractmethod
    async def count_by_status(self, status: JobStatus) -> int:
        """Count jobs by status."""
        pass
    
    @abstractmethod
    async def count_by_type(self, job_type: JobType) -> int:
        """Count jobs by type."""
        pass
    
    @abstractmethod
    async def get_job_statistics(self) -> dict:
        """Get job execution statistics."""
        pass