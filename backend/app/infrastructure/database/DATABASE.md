# Persistent Storage Implementation Guide

## 📋 Summary

You now have **ONE repository that works with BOTH databases**!

### Repository Structure

```
app/infrastructure/repositories/
├── in_memory_document_repository.py     # Temporary (no persistence)
└── sqlalchemy_document_repository.py    # Persistent (PostgreSQL + SQLite)
```

### Available Imports

```python
from app.infrastructure.repositories import (
    InMemoryDocumentRepository,      # Temporary storage
    SQLAlchemyDocumentRepository,    # Main name (recommended) ✅
    PostgresDocumentRepository,      # Alias (same class)
    SQLiteDocumentRepository,        # Alias (same class)
)

# All three persistent names are THE SAME class:
SQLAlchemyDocumentRepository == PostgresDocumentRepository == SQLiteDocumentRepository
```

---

## 🚀 Quick Start

### 1. Current Setup (SQLite + aiosqlite) ✅

Your current `.env` already has:
```bash
DATABASE_URL=sqlite+aiosqlite:///./fitvise.db
```

**Run migrations**:
```bash
uv run alembic upgrade head
```

**Use the repository**:
```python
from app.infrastructure.database import get_async_session
from app.infrastructure.repositories import SQLAlchemyDocumentRepository

async def example(session: AsyncSession = Depends(get_async_session)):
    repo = SQLAlchemyDocumentRepository(session)

    # All methods work exactly the same as InMemory!
    document = await repo.find_by_id(doc_id)
    await repo.save(document)
```

### 2. Upgrade to PostgreSQL (Production)

**Install PostgreSQL**:
```bash
# macOS
brew install postgresql
brew services start postgresql

# Create database
createdb fitvise
```

**Update .env**:
```bash
DATABASE_URL=postgresql+asyncpg://postgres:yourpass@localhost:5432/fitvise
```

**Run migrations**:
```bash
uv run alembic upgrade head
```

**No code changes needed!** The same repository works automatically.

---

## 💡 Key Benefits

### YES - aiosqlite is included! ✅

```bash
# Already added these dependencies:
✅ sqlalchemy[asyncio]  # Core ORM
✅ asyncpg              # PostgreSQL driver
✅ aiosqlite            # SQLite driver
✅ alembic              # Migrations
```

### Automatic Dialect Adaptation

The repository uses SQLAlchemy's `.with_variant()` to automatically adapt:

```python
# PostgreSQL: Uses native JSONB
document_metadata = Column(JSONB())

# SQLite: Uses JSON-as-TEXT
document_metadata = Column(JSON())

# You don't need to change anything - it just works! ✨
```

---

## 🔄 Migration Path

### From InMemory → Persistent

**Before**:
```python
repo = InMemoryDocumentRepository()
document = await repo.save(doc)
```

**After**:
```python
from app.infrastructure.database import get_async_session
from app.infrastructure.repositories import SQLAlchemyDocumentRepository

async with async_session_maker() as session:
    repo = SQLAlchemyDocumentRepository(session)
    document = await repo.save(doc)
```

**Or with dependency injection** (recommended):
```python
async def my_endpoint(session: AsyncSession = Depends(get_async_session)):
    repo = SQLAlchemyDocumentRepository(session)
    document = await repo.find_by_id(doc_id)
    return document
```

---

## 📊 Repository Comparison

| Feature | InMemory | SQLAlchemy (SQLite) | SQLAlchemy (PostgreSQL) |
|---------|----------|---------------------|-------------------------|
| **Persistent** | ❌ No | ✅ Yes | ✅ Yes |
| **Setup** | None | File | Server |
| **aiosqlite** | N/A | ✅ Included | N/A |
| **asyncpg** | N/A | N/A | ✅ Included |
| **Same API** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Same Code** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Speed** | ⚡ Fastest | 🚀 Fast | 🚀 Fast |
| **Production** | ❌ No | ⚠️ Small | ✅ Yes |

---

## 🎯 What You Asked About

### Q: "Where is SQLite document repo?"
**A**: It's the **same repository** as PostgreSQL!

The `SQLAlchemyDocumentRepository` class automatically detects which database you're using based on your `DATABASE_URL` and adapts accordingly.

### Q: "You have added aiosqlite?"
**A**: Yes! ✅ Already added as a dependency. Check:

```bash
uv pip list | grep aiosqlite
# aiosqlite  0.20.0
```

Your current `.env` uses it:
```bash
DATABASE_URL=sqlite+aiosqlite:///./fitvise.db
              ↑↑↑↑↑↑↑↑↑↑
              SQLite async driver
```

---

## 🔧 Available Commands

```bash
# Run migrations (apply schema)
uv run alembic upgrade head

# Create new migration (after model changes)
uv run alembic revision --autogenerate -m "Description"

# Rollback last migration
uv run alembic downgrade -1

# View migration history
uv run alembic history

# Check current version
uv run alembic current
```

---

## 📝 Usage Examples

### Basic CRUD
```python
from app.infrastructure.repositories import SQLAlchemyDocumentRepository

# Save document
await repo.save(document)

# Find by ID
doc = await repo.find_by_id(document_id)

# Find by status
processed = await repo.find_by_status(DocumentStatus.PROCESSED)

# Find RAG-ready documents
ready = await repo.find_ready_for_rag()

# Delete
await repo.delete(document_id)
```

### Statistics
```python
# Get processing stats
stats = await repo.get_processing_stats()
# Returns: {"total": 100, "processed": 80, "failed": 5, "pending": 15}

# Count by status
count = await repo.count_by_status(DocumentStatus.PROCESSED)
```

### Advanced Queries
```python
# Find by categories
fitness_docs = await repo.find_by_categories(["fitness", "health"])

# Find by quality score
high_quality = await repo.find_by_quality_score_range(0.8, 1.0)

# Find needing reprocessing
needs_work = await repo.find_needing_reprocessing()
```

---

## ✨ Next Steps

1. **Run migrations** to create the database:
   ```bash
   uv run alembic upgrade head
   ```

2. **Test with SQLite** (current setup):
   ```python
   # Your .env already configured!
   DATABASE_URL=sqlite+aiosqlite:///./fitvise.db
   ```

3. **Switch to PostgreSQL** when ready for production:
   ```bash
   # Update .env
   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/fitvise

   # Run migrations again (automatically creates PostgreSQL tables)
   uv run alembic upgrade head
   ```

4. **Replace InMemory usage** in your codebase:
   - Search for `InMemoryDocumentRepository`
   - Replace with `SQLAlchemyDocumentRepository`
   - Add `session: AsyncSession = Depends(get_async_session)` parameter

---

## 🎉 Summary

✅ **aiosqlite**: Included and configured
✅ **asyncpg**: Included for PostgreSQL
✅ **SQLite repo**: Same as PostgreSQL repo
✅ **Single implementation**: Works with both databases
✅ **Auto-adaptation**: Detects database from URL
✅ **Production-ready**: Full ACID, indexing, migrations

Your persistent storage is complete and ready to use! 🚀
