# DI BACKLOG.md

## Progress
```
Epic 1 Progress: ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ (24/24 tasks completed) - 100%
Epic 2 Progress: ✅✅✅✅✅✅✅✅✅✅✅✅✅✅⬜⬜⬜⬜⬜⬜ (13/21 tasks completed) - 62%
Epic 3 Progress: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (0/23 tasks completed) - 0%
Epic 4 Progress: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (0/29 tasks completed) - 0%
Epic 5 Progress: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (0/26 tasks completed) - 0%
```

## Status Summary
```
| Status       | Count |
|--------------|-------|
| todo         | 2     |
| in_progress  | 0     |
| blocked      | 0     |
| review       | 0     |
| done         | 23    |
```

# Meta
- **Goal:** Complete migration to unified dependency injection system using dependency-injector
- **Scope:** Infrastructure, API layer, service layer, repository layer, cleanup and documentation
- **Version:** 1.1
- **Updated:** 2025-12-11

# Epics Overview

## Epic 1: DI Infrastructure Setup
### Progress
✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ (24/24 tasks completed) - 100%

### Backlog Items
- [x] DI-001 Install dependency-injector library
- [x] DI-002 Create DI directory structure
- [x] DI-003 Implement configuration providers
- [x] DI-004 Implement external service providers
- [x] DI-005 Implement repository providers
- [x] DI-006 Implement service providers
- [x] DI-007 Implement main DI container
- [x] DI-008 Implement bootstrap system
- [x] DI-009 Implement testing support
- [x] DI-010 Update application entry points
- [x] DI-011 Validate DI modules import correctly
- [x] DI-012 Validate container instantiation
- [x] DI-013 Validate test container works
- [x] DI-014 Validate application starts with DI

## Epic 2: API Layer Migration
### Progress
✅✅✅✅✅✅✅✅✅✅✅✅✅✅⬜⬜⬜⬜⬜⬜ (13/21 tasks completed) - 62%

### Backlog Items
- [x] DI-015 Update embeddings router to use DI
- [x] DI-016 Test embeddings API with DI
- [x] DI-017 Update RAG ingestion router to use DI
- [x] DI-018 Update RAG search router to use DI
- [x] DI-019 Update health check endpoints
- [x] DI-020 Update RAG data sources router to use DI
- [x] DI-021 Update RAG storage router to use DI
- [ ] DI-022 Validate API endpoints use DI dependencies
- [x] DI-023 Validate no manual service instantiation in routers
- [ ] DI-024 Test API endpoints with real and mocked dependencies
- [x] DI-025 Validate health endpoints show comprehensive service status

## Epic 3: Service Layer Migration
### Progress
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (0/23 tasks completed) - 0%

### Backlog Items
- [ ] Tasks for migrating embedding use cases, LLM use cases, domain services, and pipeline workflows to DI patterns (see detailed section)

## Epic 4: Repository Layer Migration
### Progress
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (0/29 tasks completed) - 0%

### Backlog Items
- [ ] Tasks for migrating SQL repositories, vector store repositories, external service clients, and container patterns to DI patterns (see detailed section)

## Epic 5: Cleanup and Documentation
### Progress
⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (0/26 tasks completed) - 0%

### Backlog Items
- [ ] Tasks for removing legacy patterns, updating documentation, performance validation, and team training (see detailed section)

# Backlog Items (Agent-Parseable Template)

## ID: <unique_id>
### Title
<short title>

### Epic
<Epic ID>

### Type
Feature | Bug | Improvement | Research | Task

### Priority
P1 | P2 | P3

### Status
 todo | in_progress | blocked | review | done

### Description
<clear description of intent and expected outcome>

### Acceptance Criteria
- <testable criterion 1>
- <testable criterion 2>
- <testable criterion 3>

### Inputs Required
<list inputs, API references, file paths, etc.>

### Outputs Expected
<deliverables, response schema, artifacts>

### Dependencies
- <task_id_1>
- <task_id_2>

### Notes
<any constraints or important considerations>

---

## ID: DI-001
### Title
Install dependency-injector library

### Epic
1: DI Infrastructure Setup

### Type
Task

### Priority
P1

### Status
done

### Description
Install dependency-injector==4.41.0 and add to requirements.txt

### Acceptance Criteria
- dependency-injector==4.41.0 added to requirements.txt
- Library installed successfully in environment
- Import test passes

### Inputs Required
- requirements.txt file path
- Python environment access

### Outputs Expected
- Updated requirements.txt with dependency-injector
- Successful pip install verification

### Dependencies
- None

### Notes
Foundation dependency for entire DI system

---

## ID: DI-002
### Title
Create DI directory structure

### Epic
1: DI Infrastructure Setup

### Type
Task

### Priority
P1

### Status
done

### Description
Create app/di/ directory structure with providers subdirectory and __init__.py files

### Acceptance Criteria
- app/di/ directory created
- app/di/providers/ directory created
- __init__.py files created in both directories
- Directory structure follows Python package conventions

### Inputs Required
- File system access
- mkdir command availability

### Outputs Expected
- Complete DI directory structure ready for implementation

### Dependencies
- DI-001

### Notes
Structural foundation for DI implementation

---

## ID: DI-003
### Title
Implement configuration providers

### Epic
1: DI Infrastructure Setup

### Type
Feature

### Priority
P1

### Status
done

### Description
Implement app/di/providers/config.py with Settings provider, embedding/Weaviate config providers, environment detection providers, and feature flag providers

### Acceptance Criteria
- Settings singleton provider implemented
- EmbeddingModelConfig providers for realtime/production
- WeaviateConfig providers implemented
- Environment detection providers (is_production, is_development)
- Feature flag providers
- All providers properly typed and documented

### Inputs Required
- Existing configuration classes (Settings, EmbeddingModelConfig, WeaviateConfig)
- dependency-injector providers module

### Outputs Expected
- Complete config.py with all configuration providers
- Type-safe configuration access through DI

### Dependencies
- DI-001
- DI-002

### Notes
Centralizes all configuration management

---

## ID: DI-004
### Title
Implement external service providers

### Epic
1: DI Infrastructure Setup

### Type
Feature

### Priority
P1

### Status
done

### Description
Implement app/di/providers/external.py with Weaviate client provider, sentence transformer service provider, Ollama LLM service provider, and health check providers

### Acceptance Criteria
- WeaviateClient singleton provider with config injection
- SentenceTransformerService singleton provider with async initialization
- OllamaService provider for LLM integration
- LlamaIndex embedding provider
- Health check providers for all external services
- Proper lifecycle management for async services

### Inputs Required
- External service client classes
- Configuration provider references
- dependency-injector resource decorators for async init

### Outputs Expected
- Complete external.py with all service providers
- Async service initialization support
- Health check capabilities for monitoring

### Dependencies
- DI-001
- DI-002
- DI-003

### Notes
Manages all external service dependencies

---

## ID: DI-005
### Title
Implement repository providers

### Epic
1: DI Infrastructure Setup

### Type
Feature

### Priority
P1

### Status
done

### Description
Implement app/di/providers/repositories.py with database session provider, document/data source/embedding repository providers, and repository bundle provider

### Acceptance Criteria
- Database session provider with async_sessionmaker
- DocumentRepository factory provider
- DataSourceRepository factory provider
- EmbeddingRepository factory provider
- Repository bundle provider for pipeline operations
- Proper session management and cleanup

### Inputs Required
- Repository implementation classes
- Database session management code
- Configuration dependencies

### Outputs Expected
- Complete repositories.py with all repository providers
- Centralized repository configuration
- Session management through DI

### Dependencies
- DI-001
- DI-002
- DI-003

### Notes
Standardizes data access patterns

---

## ID: DI-006
### Title
Implement service providers

### Epic
1: DI Infrastructure Setup

### Type
Feature

### Priority
P1

### Status
done

### Description
Implement app/di/providers/services.py with LLM service provider, embedding domain service provider, use case providers, pipeline workflow provider, and service health check providers

### Acceptance Criteria
- LLMService provider with external service dependencies
- EmbeddingDomainService provider
- Use case providers for all business operations
- PipelineWorkflow provider with complete dependency injection
- Service health check providers
- Proper dependency wiring between all services

### Inputs Required
- Domain service classes
- Use case implementations
- Pipeline workflow classes
- Repository and external service dependencies

### Outputs Expected
- Complete services.py with all service providers
- Fully wired business logic through DI
- Health monitoring for all services

### Dependencies
- DI-001
- DI-002
- DI-003
- DI-004
- DI-005

### Notes
Core business logic dependency management

---

## ID: DI-007
### Title
Implement main DI container

### Epic
1: DI Infrastructure Setup

### Type
Feature

### Priority
P1

### Status
done

### Description
Implement app/di/container.py with main FitviseContainer class that wires all provider groups and provides convenience shortcuts for FastAPI integration

### Acceptance Criteria
- FitviseContainer class with DeclarativeContainer
- All provider groups wired together (config, external, repositories, services)
- Convenience shortcuts for common services
- Type hints for FastAPI integration
- Lifecycle management setup
- Container initialization patterns

### Inputs Required
- All provider modules
- dependency-injector containers module
- FastAPI dependency patterns

### Outputs Expected
- Complete container.py with unified dependency graph
- Easy FastAPI integration points
- Type-safe service access

### Dependencies
- DI-003
- DI-004
- DI-005
- DI-006

### Notes
Central dependency injection hub

---

## ID: DI-008
### Title
Implement bootstrap system

### Epic
1: DI Infrastructure Setup

### Type
Feature

### Priority
P1

### Status
done

### Description
Implement app/di/bootstrap.py with application creation functions, middleware configuration, route configuration, exception handlers, and health check endpoints

### Acceptance Criteria
- FastAPI application creation with DI wiring
- CORS and middleware configuration
- Route configuration helpers
- Exception handler integration
- Health check endpoints with DI services
- Environment-specific bootstrap functions
- Legacy compatibility support

### Inputs Required
- FastAPI app patterns
- Container reference
- Configuration and service providers

### Outputs Expected
- Complete bootstrap.py with application lifecycle management
- Easy FastAPI app creation patterns
- Health monitoring integration

### Dependencies
- DI-007

### Notes
Application startup and lifecycle management

---

## ID: DI-009
### Title
Implement testing support

### Epic
1: DI Infrastructure Setup

### Type
Feature

### Priority
P1

### Status
done

### Description
Implement app/di/testing.py with mock providers for all services, test container implementation, integration test container, override utilities, and pytest fixtures

### Acceptance Criteria
- Mock providers for all services and configurations
- Test container with full mocking
- Integration test container with real services
- Override utilities for selective mocking
- Pytest fixtures for easy testing
- Test isolation and cleanup utilities

### Inputs Required
- unittest.mock for mocking
- pytest fixtures
- Container override patterns
- Test configuration needs

### Outputs Expected
- Complete testing.py with comprehensive test support
- Easy dependency mocking for unit tests
- Integration test capabilities

### Dependencies
- DI-007

### Notes
Enables comprehensive testing with DI

---

## ID: DI-010
### Title
Update application entry points

### Epic
1: DI Infrastructure Setup

### Type
Feature

### Priority
P1

### Status
done

### Description
Update app/main.py and run.py with DI integration, legacy compatibility during transition, and migration status endpoint

### Acceptance Criteria
- app/main.py updated with DI-managed application lifecycle
- run.py updated with bootstrap pattern
- Legacy compatibility maintained during transition
- Migration status endpoint added
- Health checks integrated with DI
- Environment-specific configurations supported

### Inputs Required
- Existing app/main.py and run.py
- Bootstrap system from DI-008
- Container reference

### Outputs Expected
- Updated app/main.py with DI integration
- Updated run.py with bootstrap patterns
- Migration monitoring endpoint

### Dependencies
- DI-008
- DI-007

### Notes
Production application integration point

---

## ID: DI-011
### Title
Validate DI modules import correctly

### Epic
1: DI Infrastructure Setup

### Type
Testing

### Priority
P1

### Status
done

### Description
Validate that all DI modules can be imported without errors and container can be instantiated

### Acceptance Criteria
- from app.di import container, bootstrap imports successfully
- No syntax or import errors
- All dependencies resolve correctly
- Container instantiation works

### Inputs Required
- Python interpreter
- DI module structure

### Outputs Expected
- Successful import validation
- Error-free module loading

### Dependencies
- DI-010

### Notes
Basic validation of DI implementation

---

## ID: DI-012
### Title
Validate container instantiation

### Epic
1: DI Infrastructure Setup

### Type
Testing

### Priority
P1

### Status
done

### Description
Validate that DI container can be instantiated and services can be retrieved

### Acceptance Criteria
- container.settings() works without exceptions
- container.llm_service() returns proper service
- All major service providers function correctly
- No circular dependency errors

### Inputs Required
- Container implementation
- Service provider classes

### Outputs Expected
- Working container with service access
- Validated dependency resolution

### Dependencies
- DI-011

### Notes
Core functionality validation

---

## ID: DI-013
### Title
Validate test container works

### Epic
1: DI Infrastructure Setup

### Type
Testing

### Priority
P1

### Status
done

### Description
Validate that test container with mocked providers can be created and function correctly

### Acceptance Criteria
- create_test_container() creates container with mocked providers
- Mocked services return expected mock responses
- Container override mechanism works
- Test isolation maintained

### Inputs Required
- Testing module from DI-009
- Mock validation patterns

### Outputs Expected
- Working test container
- Validated mocking capabilities

### Dependencies
- DI-009
- DI-012

### Notes
Testing infrastructure validation

---

## ID: DI-014
### Title
Validate application starts with DI

### Epic
1: DI Infrastructure Setup

### Type
Testing

### Priority
P1

### Status
done

### Description
Validate that application starts successfully with DI integration

### Acceptance Criteria
- python run.py starts application successfully
- No startup errors or missing dependencies
- Health endpoints accessible
- DI services properly initialized

### Inputs Required
- Updated run.py from DI-010
- Container and bootstrap system

### Outputs Expected
- Working application startup
- Validated DI integration

### Dependencies
- DI-010
- DI-013

### Notes
Production readiness validation

---

## ID: DI-015
### Title
Update embeddings router to use DI

### Epic
2: API Layer Migration

### Type
Migration

### Priority
P2

### Status
done

### Description
Update app/api/v1/embeddings/router.py to replace get_embedding_service() and get_weaviate_client() with DI dependencies

### Acceptance Criteria
- Remove get_embedding_service() function
- Remove get_weaviate_client() function
- Update all endpoint function signatures to use DI providers
- Remove manual service instantiation
- All endpoints work with DI dependencies

### Inputs Required
- Existing embeddings router
- Container service providers
- FastAPI DI patterns

### Outputs Expected
- Updated router with DI dependencies
- Cleaner endpoint implementations

### Dependencies
- DI-010

### Notes
First API module migration example

---

## ID: DI-016
### Title
Test embeddings API with DI

### Epic
2: API Layer Migration

### Type
Testing

### Priority
P2

### Status
done

### Description
Create comprehensive tests for embeddings API using DI test container

### Acceptance Criteria
- Test all embeddings endpoints with mocked dependencies
- Test with test container override
- Validate all request/response patterns
- Test error handling scenarios

### Inputs Required
- Updated embeddings router
- Testing framework
- Test container utilities

### Outputs Expected
- Comprehensive test suite
- Validated DI migration functionality

### Dependencies
- DI-015

### Notes
Validation of first migration

---

## ID: DI-017
### Title
Update RAG ingestion router to use DI

### Epic
2: API Layer Migration

### Type
Migration

### Priority
P2

### Status
done

### Description
Update app/api/v1/rag/ingestion.py to replace manual repository creation with DI pipeline workflow provider

### Acceptance Criteria
- Replace manual repository creation
- Use DI pipeline workflow provider
- Update dependency injection patterns
- Maintain all existing functionality
- Clean up manual dependency management

### Inputs Required
- Existing RAG ingestion router
- Pipeline workflow provider
- Repository providers

### Outputs Expected
- Updated router using DI patterns
- Simplified dependency management

### Dependencies
- DI-015

### Notes
Complex workflow migration

---

## ID: DI-018
### Title
Update RAG search router to use DI

### Epic
2: API Layer Migration

### Type
Migration

### Priority
P2

### Status
done

### Description
Update app/api/v1/rag/search.py to replace manual service creation with DI search use case provider

### Acceptance Criteria
- Replace manual service creation
- Use DI search use case provider
- Update endpoint signatures
- Maintain search functionality
- Remove hardcoded dependencies

### Inputs Required
- Existing RAG search router
- Search use case provider
- Service providers

### Outputs Expected
- Updated router with DI patterns
- Streamlined search functionality

### Dependencies
- DI-015

### Notes
Search functionality migration

---

## ID: DI-019
### Title
Update health check endpoints

### Epic
2: API Layer Migration

### Type
Migration

### Priority
P2

### Status
done

### Description
Replace manual health checks with DI health check providers and add comprehensive service health monitoring

### Acceptance Criteria
- Replace manual health checks
- Use DI health check providers
- Add comprehensive service health monitoring
- Monitor all critical services
- Provide detailed health status

### Inputs Required
- Existing health endpoints
- Health check providers
- Service monitoring patterns

### Outputs Expected
- Comprehensive health monitoring
- DI-based health checks

### Dependencies
- DI-010

### Notes
System health visibility

---

## ID: DI-020
### Title
Update RAG data sources router to use DI

### Epic
2: API Layer Migration

### Type
Migration

### Priority
P2

### Status
done

### Description
Update app/api/v1/rag/data_sources.py to use DI providers for repository and service dependencies

### Acceptance Criteria
- Replace manual repository access
- Use DI repository providers
- Update endpoint signatures
- Maintain data source management
- Remove hardcoded dependencies

### Inputs Required
- Existing data sources router
- Repository providers
- Data source service providers

### Outputs Expected
- Updated router with DI patterns
- Simplified data source management

### Dependencies
- DI-018

### Notes
Data management migration

---

## ID: DI-021
### Title
Update RAG storage router to use DI

### Epic
2: API Layer Migration

### Type
Migration

### Priority
P2

### Status
done

### Description
Update app/api/v1/rag/storage.py to use DI providers for storage and repository operations

### Acceptance Criteria
- Replace manual storage service creation
- Use DI storage service providers
- Update storage endpoint patterns
- Maintain storage functionality
- Remove hardcoded storage dependencies

### Inputs Required
- Existing storage router
- Storage service providers
- Repository dependencies

### Outputs Expected
- Updated router with DI patterns
- Streamlined storage operations

### Dependencies
- DI-020

### Notes
Storage management migration

---

## ID: DI-022
### Title
Validate API endpoints use DI dependencies

### Epic
2: API Layer Migration

### Type
Testing

### Priority
P2

### Status
todo

### Description
Validate that all API endpoints use DI dependencies instead of manual instantiation

### Acceptance Criteria
- No manual service instantiation in routers
- All endpoints use DI providers
- Dependency injection patterns consistent
- No legacy dependency patterns remain

### Inputs Required
- All API router files
- Container provider references
- Dependency analysis tools

### Outputs Expected
- Clean DI-based API layer
- Validated dependency patterns

### Dependencies
- DI-015
- DI-017
- DI-018
- DI-019
- DI-020
- DI-021

### Notes
API layer completion validation

---

## ID: DI-023
### Title
Validate no manual service instantiation in routers

### Epic
2: API Layer Migration

### Type
Testing

### Priority
P2

### Status
done

### Description
Comprehensive validation that no routers contain manual service instantiation

### Acceptance Criteria
- Search all router files for manual instantiation patterns
- No direct imports of service classes for instantiation
- No manual configuration creation in endpoints
- Clean separation of concerns maintained

### Inputs Required
- Router file analysis
- Pattern detection tools
- Code review checklist

### Outputs Expected
- Validated clean dependency patterns
- Documentation of dependency usage

### Dependencies
- DI-022

### Notes
Quality assurance for API layer

---

## ID: DI-024
### Title
Test API endpoints with real and mocked dependencies

### Epic
2: API Layer Migration

### Type
Testing

### Priority
P2

### Status
todo

### Description
Create comprehensive test suite for all API endpoints with both real and mocked dependencies

### Acceptance Criteria
- All endpoints tested with mocked dependencies
- All endpoints tested with real dependencies
- Integration tests cover end-to-end scenarios
- Performance tests with DI
- Error handling validation

### Inputs Required
- Complete API layer
- Test infrastructure
- Real and test containers
- Test data fixtures

### Outputs Expected
- Comprehensive API test suite
- Validated DI functionality
- Performance benchmarks

### Dependencies
- DI-016
- DI-022
- DI-023

### Notes
Complete API validation

---

## ID: DI-025
### Title
Validate health endpoints show comprehensive service status

### Epic
2: API Layer Migration

### Type
Testing

### Priority
P2

### Status
done

### Description
Validate that health endpoints provide comprehensive service status using DI providers

### Acceptance Criteria
- Health endpoints monitor all critical services
- Service status reflects real health conditions
- Detailed diagnostics available
- Performance metrics included
- Error scenarios handled gracefully

### Inputs Required
- Health check providers
- Service monitoring implementation
- Health endpoint testing

### Outputs Expected
- Working comprehensive health monitoring
- Validated service visibility

### Dependencies
- DI-019

### Notes
System observability validation

---

# Remaining Epics and Tasks

## Epic 3: Service Layer Migration (Phase 3) - 0/23 tasks
Tasks for migrating embedding use cases, LLM use cases, domain services, and pipeline workflows to DI patterns.

## Epic 4: Repository Layer Migration (Phase 4) - 0/29 tasks
Tasks for migrating SQL repositories, vector store repositories, external service clients, and container patterns to DI patterns.

## Epic 5: Cleanup and Documentation (Phase 5) - 0/26 tasks
Tasks for removing legacy patterns, updating documentation, performance validation, and team training.

# How to Update This Backlog
- Update progress bars with `⬜` and `✅`
- Update Status Summary table
- Add tasks following the template
- Mark dependencies with proper task IDs
