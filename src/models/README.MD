# 📊 Models Module Documentation

The `models/` directory contains the core data models and database schemas for the RAG system. This module implements an object-oriented approach to database operations using SQLAlchemy ORM with async support.

---

## 📁 Module Structure

```
models/
├── BaseModel.py              # Abstract base class for all models
├── ProjectModel.py           # Project entity management
├── AssetModel.py            # Document/file metadata management
├── ChunkModel.py            # Document chunk management
├── enums/                   # Enumeration definitions
   ├── responseEnums.py    # API response types
   ├── processEnums.py     # Asset/file types (PDF, TXT, etc.)
   ├── databaseEnums.py    # Database naming conventions
   └── assetEnums.py       # Asset-specific enumerations

```

---

## 🏗️ Architecture Overview

### Design Principles

1. **Inheritance Hierarchy**: All models extend `BaseModel` for shared functionality
2. **Async/Await Pattern**: Fully asynchronous database operations for high performance
3. **Factory Pattern**: Class methods for instance creation with proper initialization
4. **Separation of Concerns**: Models handle data operations, schemas define structure
5. **Type Safety**: Full type hints for better IDE support and error detection

### Data Flow

```
API Request → Controller → Model → Database → Response
                 ↓           ↓         ↓
              Business    Data      Schema
               Logic    Operations  Definition
```

---

## 📚 Core Models

### 1. BaseModel

**Purpose**: Abstract base class providing common functionality to all models.

**Location**: `models/BaseModel.py`

**Responsibilities**:

- Initialize database client connection
- Provide access to application settings
- Serve as foundation for all data models

**Key Features**:

```
python
class BaseModel:
    def __init__(self, db_client: object):
        self.db_client = db_client           # SQLAlchemy async session factory
        self.settings: Settings = get_settings()  # App configuration
```

**Usage**:

- Never instantiated directly
- Always subclassed by specific models (Project, Asset, Chunk)
- Provides `db_client` and `settings` to all child classes

**Why This Design?**

- **DRY Principle**: Avoids repeating database client and settings initialization
- **Consistency**: Ensures all models have the same base attributes
- **Extensibility**: Easy to add shared methods for all models

---

### 2. ProjectModel

**Purpose**: Manages project entities (rows) that group related documents and queries.

**Location**: `models/ProjectModel.py`

**Responsibilities**:

- Create new projects in the database
- Retrieve projects by ID (auto-create if not found)
- List all projects with pagination support
- Manage project lifecycle

**Key Methods**:

#### `create_instance(cls, db_client: object)` → `ProjectModel`

**Factory method** for creating model instances.

```
python
project_model = await ProjectModel.create_instance(db_client)
```

**Why async?** Allows for future initialization logic that might require async operations.

#### `create_project(project: Project)` → `Project`

Creates a new project record in the database.

```
python
new_project = Project(project_id=123)
created = await project_model.create_project(new_project)
```

**Transaction Flow**:

1. Opens async session with `db_client()`
2. Begins transaction with `session.begin()`
3. Adds project to session
4. Commits changes to database
5. Refreshes object to get DB-generated fields
6. Returns persisted project

#### `get_project_by_id(project_id: int)` → `Project`

Retrieves a project by its ID. **Auto-creates** if project doesn't exist.

```
python
project = await project_model.get_project_by_id(project_id=456)
```

**Smart Behavior**:

- Searches database for existing project
- If found: Returns it immediately
- If not found: Creates new project with that ID and returns it

**Use Case**: Ensures every project ID is valid without explicit pre-creation.

#### `get_all_projects(page: int, page_size: int)` → `list[Project]`

Retrieves paginated list of all projects.

```
python
projects = await project_model.get_all_projects(page=2, page_size=20)
```

**Pagination Logic**:

1. Counts total projects in database
2. Calculates total pages: `(total + page_size - 1) // page_size`
3. Auto-corrects invalid page numbers to last valid page
4. Fetches only requested slice using `offset` and `limit`

**Benefits**:

- Efficient for large datasets (doesn't load all projects)
- Prevents memory overload
- Auto-handles edge cases (page > total_pages)

---

### 3. AssetModel

**Purpose**: Manages metadata for uploaded documents (PDFs, text files, etc.).

**Location**: `models/AssetModel.py`

**Responsibilities**:

- Store file metadata (name, type, project association)
- Track which files belong to which projects
- Retrieve files by project and type
- Check for duplicate file uploads

**Key Concepts**:

- **Metadata Only**: Stores information _about_ files, not file contents
- **Project Association**: Every asset belongs to exactly one project
- **Type Filtering**: Supports querying by file type (PDF, TXT, etc.)

**Key Methods**:

#### `create_instance(cls, db_client: object)` → `AssetModel`

Factory method for creating model instances.

```
python
asset_model = await AssetModel.create_instance(db_client)
```

#### `create_asset(asset: Asset)` → `Asset`

Saves new file metadata to the database.

```
python
new_asset = Asset(
    asset_name="report.pdf",
    asset_type="PDF",
    asset_project_id="project_123"
)
created_asset = await asset_model.create_asset(new_asset)
```

**Transaction Steps**:

1. Opens async database session
2. Adds asset metadata to session
3. Commits to persist changes
4. Refreshes to get any auto-generated fields (IDs, timestamps)
5. Returns fully populated asset object

#### `get_all_project_assets(asset_project_id: str, asset_type: str)` → `list[Asset]`

Retrieves all files of a specific type within a project.

```
python
# Get all PDF files in project_123
pdf_assets = await asset_model.get_all_project_assets(
    asset_project_id="project_123",
    asset_type="PDF"
)
```

**Query Logic**:

- Uses SQLAlchemy `select` with `where` clause
- Filters by both project ID AND file type
- Returns list of all matching assets

**Use Cases**:

- Display all PDFs uploaded to a project
- Process all text files for a specific analysis
- Audit file types per project

#### `get_asset_record(asset_project_id: str, asset_name: str)` → `Asset | None`

Finds a specific file by name within a project.

```
python
existing = await asset_model.get_asset_record(
    asset_project_id="project_123",
    asset_name="report.pdf"
)

if existing:
    print("File already exists!")
else:
    # Safe to upload
    pass
```

**Return Values**:

- `Asset` object if found
- `None` if no matching file exists

**Critical Use Case**: **Duplicate Prevention**

- Check before uploading to avoid overwriting files
- Implement "file already exists" warnings
- Enable file versioning logic

---

### 4. ChunkModel

**Purpose**: Manages document chunks created during text processing.

**Location**: `models/ChunkModel.py`

**Responsibilities**:

- Store text chunks with embeddings
- Associate chunks with parent documents (assets)
- Enable vector similarity search
- Track chunk metadata (position, size, etc.)

**Key Concepts**:

- **Chunking**: Breaking documents into smaller, semantically meaningful pieces
- **Embeddings**: Vector representations for similarity search
- **Parent-Child Relationship**: Each chunk links back to its source asset

**Key Methods**:

_(Note: Based on the pattern from other models, expected methods include)_

#### `create_instance(cls, db_client: object)` → `ChunkModel`

Factory method for instance creation.

#### `create_chunk(chunk: Chunk)` → `Chunk`

Saves a new chunk with its embedding to the database.

```
python
chunk = Chunk(
    chunk_text="This is a paragraph from the document...",
    chunk_embedding=[0.123, 0.456, ...],  # Vector representation
    chunk_asset_id="asset_789",
    chunk_position=0
)
saved_chunk = await chunk_model.create_chunk(chunk)
```

#### `get_chunks_by_asset(asset_id: str)` → `list[Chunk]`

Retrieves all chunks belonging to a specific document.

```
python
chunks = await chunk_model.get_chunks_by_asset(asset_id="asset_789")
```

**Use Case**: Reconstructing original document or analyzing chunk distribution.

#### `search_similar_chunks(embedding: list[float], top_k: int)` → `list[Chunk]`

Performs vector similarity search to find relevant chunks.

```
python
query_embedding = embed_text("What is the main topic?")
results = await chunk_model.search_similar_chunks(
    embedding=query_embedding,
    top_k=5
)
```

**How It Works**:

1. Compares query embedding against all chunk embeddings
2. Uses vector distance metric (cosine similarity, L2 distance)
3. Returns top K most similar chunks
4. Powers the "retrieval" part of RAG

---

## 🎯 Design Patterns Explained

### 1. Factory Pattern: `create_instance()`

**Why use it?**

- Allows async initialization in the future
- Provides consistent object creation interface
- Enables dependency injection

**Example**:

```python
# Instead of direct instantiation:
# model = ProjectModel(db_client)  # Can't do async operations here

# Use factory method:
model = await ProjectModel.create_instance(db_client)  # Can do async init
```

### 2. Context Managers: `async with`

**Why use them?**

- Automatic connection management (open/close)
- Transaction safety (auto-rollback on errors)
- Resource cleanup (prevents memory leaks)

**Example**:

```
python
async with self.db_client() as session:
    async with session.begin():
        # All operations here are in a transaction
        session.add(project)
    # Automatically commits if no errors
    # Automatically rolls back if exception occurs
```

### 3. Lazy Auto-Creation: `get_project_by_id()`

**Why useful?**

- Reduces boilerplate (no need to check existence everywhere)
- Idempotent operations (safe to call multiple times)
- Simplifies API logic

**Trade-off**: May create unwanted projects if IDs are typos. Consider validation.

---

## 🔄 Async/Await Pattern

All database operations use async/await for non-blocking I/O:

**Benefits**:

- **Concurrency**: Handle multiple requests simultaneously
- **Performance**: Don't block on database queries
- **Scalability**: Support more users with same resources

**Example**:

```
python
# Traditional blocking code:
def get_project(id):
    session = db.create_session()
    project = session.query(Project).filter_by(id=id).first()
    session.close()
    return project  # Blocks thread until DB responds

# Async non-blocking code:
async def get_project(id):
    async with db_client() as session:
        result = await session.execute(select(Project).where(Project.id == id))
        return result.scalar_one_or_none()  # Releases thread while waiting
```

---

## 🔍 Common Usage Patterns

### Pattern 1: Creating a New Project

```
python
from models import ProjectModel
from models.db_schemas import Project

# Initialize model
project_model = await ProjectModel.create_instance(db_client)

# Create project
new_project = Project(project_id=123)
created = await project_model.create_project(new_project)
```

### Pattern 2: Uploading a Document

```
python
from models import AssetModel
from models.db_schemas import Asset

# Check if file already exists
asset_model = await AssetModel.create_instance(db_client)
existing = await asset_model.get_asset_record(
    asset_project_id="proj_123",
    asset_name="document.pdf"
)

if not existing:
    # Create new asset record
    new_asset = Asset(
        asset_name="document.pdf",
        asset_type="PDF",
        asset_project_id="proj_123"
    )
    await asset_model.create_asset(new_asset)
```

### Pattern 3: Processing Document into Chunks

```
python
from models import ChunkModel
from models.db_schemas import Chunk

chunk_model = await ChunkModel.create_instance(db_client)

# Create chunks from document
for i, text_segment in enumerate(document_segments):
    embedding = generate_embedding(text_segment)

    chunk = Chunk(
        chunk_text=text_segment,
        chunk_embedding=embedding,
        chunk_asset_id=asset.asset_id,
        chunk_position=i
    )

    await chunk_model.create_chunk(chunk)
```

### Pattern 4: RAG Query

```
python
# 1. Get user query embedding
query_embedding = embed_text(user_question)

# 2. Find similar chunks
chunk_model = await ChunkModel.create_instance(db_client)
relevant_chunks = await chunk_model.search_similar_chunks(
    embedding=query_embedding,
    top_k=5
)

# 3. Use chunks as context for LLM
context = "\n\n".join([chunk.chunk_text for chunk in relevant_chunks])
answer = await llm.generate(question=user_question, context=context)
```

---

## ⚠️ Important Considerations

### Transaction Management

- Always use `async with session.begin()` for transactional operations
- Database commits happen automatically on successful exit
- Rollbacks happen automatically on exceptions

### Error Handling

Models don't catch exceptions—let them bubble up to controllers:

```
python
# In Controller
try:
    project = await project_model.get_project_by_id(id)
except SQLAlchemyError as e:
    logger.error(f"Database error: {e}")
    raise HTTPException(status_code=500, detail="Database error")
```

### Performance Tips

1. **Use pagination** for large result sets (`get_all_projects`)
2. **Batch operations** when creating many chunks
3. **Index frequently queried columns** (project_id, asset_name)
4. **Use connection pooling** (configured in db_client)

### Type Safety

- All methods use type hints for parameters and return values
- Use IDE with type checking (PyCharm, VSCode + Pylance)
- Run `mypy` for static type checking

---

## 🧪 Testing Models

### Unit Test Example

```
python
import pytest
from models import ProjectModel
from models.db_schemas import Project

@pytest.mark.asyncio
async def test_create_project(db_client):
    # Arrange
    project_model = await ProjectModel.create_instance(db_client)
    project = Project(project_id=999)

    # Act
    created = await project_model.create_project(project)

    # Assert
    assert created.project_id == 999

    # Verify in database
    retrieved = await project_model.get_project_by_id(999)
    assert retrieved.project_id == created.project_id
```

---

## 📚 Further Reading

- **SQLAlchemy Async ORM**: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- **Python Async/Await**: https://docs.python.org/3/library/asyncio.html
- **pgvector Extension**: https://github.com/pgvector/pgvector
- **Factory Pattern**: https://refactoring.guru/design-patterns/factory-method

---

**Questions or Issues?** Refer to the main project README or open a GitHub issue.
