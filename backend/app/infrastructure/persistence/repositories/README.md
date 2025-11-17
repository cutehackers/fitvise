# Document Repository Implementations

This directory contains three implementations of the `DocumentRepository` interface:

## 1. InMemoryDocumentRepository ⚡

**Use Case**: Testing, prototyping, temporary storage

**Location**: `in_memory_document_repository.py`

**Characteristics**:
- ✅ Zero configuration required
- ✅ Fast for small datasets (<1000 documents)
- ❌ Data lost on restart
- ❌ No persistence
- ❌ Memory-limited

**Usage**:
```python
from app.infrastructure.repositories import InMemoryDocumentRepository

repo = InMemoryDocumentRepository()
await repo.save(document)
```

---

## 2. SQLAlchemyDocumentRepository 🗄️

**Use Case**: Production and development with persistent storage

**Location**: `sqlalchemy_document_repository.py`

**Characteristics**:
- ✅ Persistent storage (survives restarts)
- ✅ Works with **both PostgreSQL AND SQLite**
- ✅ Full async support
- ✅ Automatic database dialect adaptation
- ✅ Production-ready with proper indexing
- ✅ ACID transactions

### Database Support

#### SQLite (Development)
```bash
# .env
DATABASE_URL=sqlite+aiosqlite:///./fitvise.db
```

**Pros**:
- Zero configuration
- Single file database
- Perfect for local development
- Easy to reset/backup

**Cons**:
- Limited concurrent writes
- No native UUID/array types (emulated via JSON)
- Less powerful JSON queries

#### PostgreSQL (Production)
```bash
# .env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/fitvise
```

**Pros**:
- Native JSONB, UUID, array types
- Excellent concurrent performance
- Powerful JSON queries with GIN indexes
- Full-text search
- Production-proven at scale

**Cons**:
- Requires separate database server
- More complex setup

### Usage

```python
from app.infrastructure.database import get_async_session
from app.infrastructure.repositories import SQLAlchemyDocumentRepository

# Using dependency injection (recommended)
async def my_function(session: AsyncSession = Depends(get_async_session)):
    repo = SQLAlchemyDocumentRepository(session)
    document = await repo.find_by_id(doc_id)
    return document

# Direct usage
async with async_session_maker() as session:
    repo = SQLAlchemyDocumentRepository(session)
    documents = await repo.find_by_status(DocumentStatus.PROCESSED)
```

### Backward-Compatible Aliases

For convenience and clarity, the following aliases are available:

```python
from app.infrastructure.repositories import (
    SQLAlchemyDocumentRepository,  # Main name (recommended)
    PostgresDocumentRepository,    # Alias (backward compatible)
    SQLiteDocumentRepository,      # Alias (for clarity)
)

# All three are the SAME implementation - just different names!
repo1 = SQLAlchemyDocumentRepository(session)
repo2 = PostgresDocumentRepository(session)   # Same class
repo3 = SQLiteDocumentRepository(session)     # Same class
```

---

## Repository Method Reference

All repository implementations provide the same interface:

### CRUD Operations
- `save(document)` - Create or update document
- `find_by_id(document_id)` - Get document by UUID
- `delete(document_id)` - Remove document
- `delete_by_source_id(source_id)` - Remove all documents from source

### Query Methods
- `find_by_source_id(source_id)` - Get all documents from source
- `find_by_file_path(file_path)` - Find by file path
- `find_by_status(status)` - Filter by processing status
- `find_by_format(format)` - Filter by document format
- `find_by_categories(categories)` - Filter by categories (any match)
- `find_by_quality_score_range(min, max)` - Filter by quality score

### Business Logic Queries
- `find_processed_documents()` - Get successfully processed docs
- `find_failed_documents()` - Get failed documents
- `find_ready_for_rag()` - Get docs ready for RAG (processed + chunks + embeddings)
- `find_needing_reprocessing()` - Get docs that need reprocessing

### Statistics
- `count_all()` - Total document count
- `count_by_source_id(source_id)` - Count by source
- `count_by_status(status)` - Count by status
- `count_by_format(format)` - Count by format
- `get_processing_stats()` - Get comprehensive processing statistics

---

## Migration Guide

### From InMemory to Persistent Storage

**Step 1**: Update your dependency injection:

```python
# Before (in-memory)
def get_document_repository() -> DocumentRepository:
    return InMemoryDocumentRepository()

# After (persistent)
async def get_document_repository(
    session: AsyncSession = Depends(get_async_session)
) -> DocumentRepository:
    return SQLAlchemyDocumentRepository(session)
```

**Step 2**: Run database migrations:

```bash
uv run alembic upgrade head
```

**Step 3**: Test with SQLite first, then switch to PostgreSQL for production.

---

## Database Setup

### SQLite (Development)

**No setup needed!** Just set the environment variable:

```bash
# .env
DATABASE_URL=sqlite+aiosqlite:///./fitvise.db
```

### PostgreSQL (Production)

**1. Install PostgreSQL**:
```bash
# macOS
brew install postgresql

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=yourpass postgres:15
```

**2. Create database**:
```bash
createdb fitvise
```

**3. Update .env**:
```bash
DATABASE_URL=postgresql+asyncpg://postgres:yourpass@localhost:5432/fitvise
```

**4. Run migrations**:
```bash
uv run alembic upgrade head
```

---

## Performance Comparison

| Feature | InMemory | SQLite | PostgreSQL |
|---------|----------|--------|------------|
| Setup | None | File | Server |
| Persistence | ❌ No | ✅ Yes | ✅ Yes |
| ACID | ❌ No | ✅ Yes | ✅ Yes |
| Concurrent Writes | ✅ Fast | ⚠️ Limited | ✅ Excellent |
| JSON Queries | ⚠️ Manual | ⚠️ Basic | ✅ Advanced |
| Full-text Search | ❌ No | ⚠️ Basic | ✅ Advanced |
| Scalability | ❌ Memory | ⚠️ Single file | ✅ Distributed |
| Backup | ❌ N/A | ✅ Copy file | ✅ pg_dump |
| Production Ready | ❌ No | ⚠️ Small scale | ✅ Yes |

---

## Recommended Usage

- **Local Development**: SQLite (`sqlite+aiosqlite://`)
- **Testing**: InMemory or SQLite
- **Production**: PostgreSQL (`postgresql+asyncpg://`)
- **Prototyping**: InMemory

---

## Additional Resources

- [SQLAlchemy Async Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Alembic Migrations Guide](https://alembic.sqlalchemy.org/en/latest/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
