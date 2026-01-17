# 🎮 Controllers Module Documentation

The `controllers/` directory contains the business logic layer of the RAG system. Controllers orchestrate data flow between API routes, models, and external services (LLMs, vector databases, evaluation frameworks). Each controller is specialized for a specific domain of functionality.

---

## 📁 Module Structure

```
controllers/
├── BaseController.py           # Abstract base with shared utilities
├── ProjectController.py        # Project directory management
├── DataController.py           # File upload and validation
├── ProcessController.py        # Document parsing and chunking
├── NLPController.py            # RAG pipeline orchestration
└── EvaluationController.py     # RAG evaluation with Ragas
```

---

## 🏗️ Architecture Overview

### Controller Hierarchy

```
BaseController (foundation)
    ↓
├── ProjectController (file system operations)
├── DataController (file validation)
├── ProcessController (document processing)
├── NLPController (RAG queries)
└── EvaluationController (RAG evaluation)
```

### Responsibility Layers

```
API Routes → Controllers → Models → Database
              ↓            ↓
         Business      Data Access
          Logic        Operations
```

**Key Principles**:

1. **Single Responsibility**: Each controller handles one domain
2. **Dependency Injection**: External services passed via constructor
3. **Async First**: All I/O operations are asynchronous
4. **No Direct DB Access**: Controllers use Models for database operations
5. **Provider Agnostic**: Works with any LLM/VectorDB through interfaces

---

## 📚 Core Controllers

### 1. BaseController

**Purpose**: Foundation class providing shared utilities and configuration to all controllers.

**Location**: `controllers/BaseController.py`

**Responsibilities**:

- Load application settings
- Provide base directory paths
- Generate random strings for unique identifiers

**Key Attributes**:

```
python
class BaseController:
    def __init__(self):
        self.app_settings: Settings          # Application configuration
        self.base_dir: str                   # Project root directory
        self.files_dir: str                  # Base directory for all uploaded files
```

**Key Methods**:

#### `generate_random_string(length: int = 8)` → `str`

Generates cryptographically random strings for unique file identifiers.

```
python
random_id = controller.generate_random_string(12)
# Output: "aB3xK9mQ2pL1"
```

**Use Cases**:

- Creating unique file names to prevent collisions
- Generating temporary identifiers
- Creating random suffixes for uploaded files

**Implementation Details**:

- Uses `string.ascii_letters` (a-z, A-Z) + `string.digits` (0-9)
- Length parameter controls output string size
- Default length is 8 characters

**Why This Design?**

- Centralized utility prevents code duplication
- All controllers inherit the same random generation logic
- Easy to change algorithm in one place (e.g., switch to UUID)

---

### 2. ProjectController

**Purpose**: Manages project-specific file system operations.

**Location**: `controllers/ProjectController.py`

**Responsibilities**:

- Create and manage project directories
- Provide project file paths
- Ensure directory structure exists

**Key Concepts**:

- **Project Isolation**: Each project gets its own directory
- **Lazy Creation**: Directories created on first access
- **Path Safety**: Uses `os.path.join` for cross-platform compatibility

**Key Methods**:

#### `get_project_files_dir(project_id: str)` → `str`

Returns the directory path for a project's files, creating it if it doesn't exist.

```
python
project_controller = ProjectController()
project_path = project_controller.get_project_files_dir(project_id="project_123")
# Returns: "/path/to/rag-project/src/assets/files/project_123"
```

**Behavior**:

1. Constructs path: `{base_files_dir}/{project_id}`
2. Checks if directory exists
3. If not exists: Creates directory (including parent directories)
4. Returns absolute path

**Directory Structure Created**:

```
assets/files/
├── project_123/
│   ├── file1.pdf
│   └── file2.txt
├── project_456/
│   └── file3.pdf
└── project_789/
    ├── file4.txt
    └── file5.pdf
```

**Thread Safety**: Uses `os.makedirs()` which is safe for concurrent calls.

**Use Cases**:

- Before saving uploaded files
- When listing project files
- During project cleanup operations

---

### 3. DataController

**Purpose**: Handles file upload validation and file system operations.

**Location**: `controllers/DataController.py`

**Responsibilities**:

- Validate uploaded files (type, size)
- Sanitize file names
- Generate unique file paths
- Prevent file name collisions

**Key Attributes**:

```
python
class DataController(BaseController):
    def __init__(self):
        self.scale_mb = 1024 * 1024      # Convert MB to bytes
        self.scale_kb = 1024              # Convert KB to bytes
```

**Key Methods**:

#### `validate_file(file: UploadFile)` → `(bool, str)`

Validates uploaded files against security and size constraints.

```
python
data_controller = DataController()
is_valid, message = data_controller.validate_file(uploaded_file)

if is_valid:
    # Proceed with upload
    pass
else:
    # Return error: message contains reason
    raise HTTPException(status_code=400, detail=message)
```

**Validation Checks**:

1. **Content Type**: Must be in `FILE_ALLOWED_TYPES` from settings
2. **File Size**: Must be under `FILE_MAX_SIZE` MB limit

**Return Values**:

- `(True, "FILE_VALIDATED_SUCCESS")` - File is valid
- `(False, "FILE_TYPE_NOT_SUPPORTED")` - Invalid file type
- `(False, "FILE_SIZE_EXCEEDED")` - File too large

**Configuration** (from `.env`):

```env
FILE_ALLOWED_TYPES=["application/pdf", "text/plain"]
FILE_MAX_SIZE=50  # MB
```

#### `get_clean_file_name(orig_file_name: str)` → `str`

Sanitizes file names to prevent security issues and file system errors.

```
python
cleaned = data_controller.get_clean_file_name("My Report (2024).pdf")
# Output: "My_Report_2024.pdf"
```

**Cleaning Steps**:

1. Trim whitespace: `"  file.pdf  "` → `"file.pdf"`
2. Remove special characters: Keep only alphanumeric, underscore, and dot
3. Replace spaces with underscores: `"my file.pdf"` → `"my_file.pdf"`

**Security Benefits**:

- Prevents path traversal attacks (`../../etc/passwd`)
- Removes shell metacharacters (`file; rm -rf /`)
- Ensures cross-platform compatibility

**Allowed Characters**: `a-z`, `A-Z`, `0-9`, `_`, `.`

#### `generate_unique_file_path(project_id: str, file_name: str)` → `(str, str)`

Creates a unique file path and name to prevent overwrites.

```
python
file_path, file_id = data_controller.generate_unique_file_path(
    project_id="project_123",
    file_name="report.pdf"
)
# file_path: "/path/to/project_123/aB3xK9mQ2pL1_report.pdf"
# file_id: "aB3xK9mQ2pL1_report.pdf"
```

**Process**:

1. Clean the original file name
2. Generate 12-character random suffix
3. Construct path: `{project_dir}/{random_suffix}_{clean_name}`
4. Check if path exists
5. If exists: Generate new random suffix and retry
6. Return tuple: (full_path, file_id)

**Return Values**:

- `file_path`: Absolute path where file will be saved
- `file_id`: Unique identifier for database storage (random_suffix + original_name)

**Collision Prevention**:

- Infinite loop continues until unique path found
- Probability of collision: ~1 in 62^12 (extremely rare)

**Usage Pattern**:

```
python
# In upload endpoint
file_path, file_id = data_controller.generate_unique_file_path(
    project_id=str(project.project_id),
    file_name=uploaded_file.filename
)

# Save file
with open(file_path, "wb") as f:
    f.write(await uploaded_file.read())

# Store metadata
asset = Asset(asset_id=file_id, asset_name=file_id, ...)
await asset_model.create_asset(asset)
```

---

### 4. ProcessController

**Purpose**: Parses documents and splits them into chunks for embedding.

**Location**: `controllers/ProcessController.py`

**Responsibilities**:

- Load documents from various formats (PDF, TXT)
- Split documents into semantic chunks
- Extract metadata from documents
- Handle different file types with appropriate parsers

**Dependencies**:

- **LangChain**: Document loaders and text splitters
- **PyMuPDF**: PDF parsing library

**Key Attributes**:

```
python
class ProcessController(BaseController):
    def __init__(self, project_id: str):
        self.project_id: str           # Current project context
        self.project_path: str          # Path to project's files
```

**Key Methods**:

#### `get_file_extension(file_id: str)` → `str`

Extracts file extension from file identifier.

```
python
ext = process_controller.get_file_extension("aB3xK9_report.pdf")
# Output: ".pdf"
```

**Why Needed?**: Determines which loader to use for parsing.

#### `get_file_loader(file_id: str)` → `Loader | None`

Returns appropriate document loader based on file type.

```
python
loader = process_controller.get_file_loader("aB3xK9_document.pdf")
# Returns: PyMuPDFLoader instance

loader = process_controller.get_file_loader("xyz123_notes.txt")
# Returns: TextLoader instance
```

**Supported Formats**:

- **PDF**: Uses `PyMuPDFLoader` for robust PDF parsing
- **TXT**: Uses `TextLoader` with UTF-8 encoding

**Extension → Loader Mapping**:

```
python
ProcessingEnum.TXT.value (".txt")  → TextLoader
ProcessingEnum.PDF.value (".pdf")  → PyMuPDFLoader
```

**Return Value**: `None` if file type is unsupported

**Why PyMuPDF?**

- Fast and memory-efficient
- Handles complex PDFs (forms, annotations)
- Extracts text with layout preservation

#### `get_file_content(file_id: str)` → `list`

Loads and parses a document into structured format.

```
python
content = process_controller.get_file_content("aB3xK9_report.pdf")
# Returns: List of Document objects with page_content and metadata
```

**Output Structure**:

```
python
[
    Document(
        page_content="Text from page 1...",
        metadata={"page": 0, "source": "path/to/file.pdf"}
    ),
    Document(
        page_content="Text from page 2...",
        metadata={"page": 1, "source": "path/to/file.pdf"}
    ),
    ...
]
```

**Process Flow**:

1. Gets appropriate loader for file type
2. Calls `loader.load()` to parse document
3. Returns list of Document objects

#### `process_file_content(file_content: list, file_id: str, chunk_size: int, overlap_size: int)` → `list`

Splits document content into chunks suitable for embedding.

```
python
chunks = process_controller.process_file_content(
    file_content=loaded_docs,
    file_id="report.pdf",
    chunk_size=1000,      # Characters per chunk
    overlap_size=200      # Character overlap between chunks
)
```

**Chunking Strategy**: Uses `RecursiveCharacterTextSplitter`

**Why Recursive?**

- Tries to split on natural boundaries (paragraphs, sentences)
- Falls back to character splitting if needed
- Preserves semantic coherence

**Parameters**:

- `chunk_size`: Target chunk length in characters
- `overlap_size`: Overlap between consecutive chunks
- `length_function`: `len` (counts characters)

**Why Overlap?**

- Prevents information loss at chunk boundaries
- Improves retrieval quality (partial matches)
- Typical overlap: 10-20% of chunk size

**Process Steps**:

1. Extract text content from Document objects
2. Extract metadata from Document objects
3. Create text splitter with specified parameters
4. Split texts while preserving metadata
5. Return list of chunk Document objects

**Output Format**:

```
python
[
    Document(
        page_content="Chunk 1 text...",
        metadata={"page": 0, "source": "file.pdf", "chunk_id": 0}
    ),
    Document(
        page_content="Chunk 2 text... (with overlap from Chunk 1)",
        metadata={"page": 0, "source": "file.pdf", "chunk_id": 1}
    ),
    ...
]
```

**Recommended Chunk Sizes**:

- **Small documents** (articles): 500-800 characters
- **Medium documents** (reports): 800-1200 characters
- **Large documents** (books): 1000-1500 characters

**Performance Considerations**:

- Smaller chunks: More precise retrieval, more storage
- Larger chunks: More context, less storage, slower search

---

### 5. NLPController

**Purpose**: Orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline.

**Location**: `controllers/NLPController.py`

**Responsibilities**:

- Manage vector database collections
- Index document chunks with embeddings
- Perform semantic search
- Generate answers using retrieved context
- Coordinate LLM and embedding providers

**Dependencies** (Injected):

- `vectordb_client`: Vector database interface (pgvector, Pinecone, etc.)
- `generation_client`: LLM interface (Gemini, OpenAI, etc.)
- `embedding_client`: Embedding model interface
- `template_parser`: Prompt template manager

**Key Concepts**:

- **Collection**: Vector database index for a project
- **Embedding**: Vector representation of text
- **Semantic Search**: Find similar vectors in high-dimensional space
- **Reranking**: Re-score search results for better relevance

**Key Methods**:

#### `create_collection_name(project_id: str)` → `str`

Generates standardized collection name for vector database.

```
python
collection_name = nlp_controller.create_collection_name(project_id="123")
# Output: "collection_768_123"
```

**Format**: `collection_{vector_size}_{project_id}`

**Why Include Vector Size?**

- Prevents mixing embeddings of different dimensions
- Allows migration to different embedding models
- Enables A/B testing with multiple embedding sizes

#### `reset_vector_db_collection(project: Project)` → `bool`

Deletes all vectors in a project's collection (fresh start).

```
python
success = await nlp_controller.reset_vector_db_collection(project)
```

**Use Cases**:

- Re-indexing documents after changes
- Switching embedding models
- Clearing test data

**Warning**: This is a destructive operation—all embeddings are deleted.

#### `get_vector_db_collection_info(project: Project)` → `dict`

Retrieves metadata about a collection (size, dimensions, count).

```
python
info = await nlp_controller.get_vector_db_collection_info(project)
# Returns: {
#     "name": "collection_768_123",
#     "dimension": 768,
#     "vectors_count": 1500,
#     "indexed": true
# }
```

**Use Cases**:

- Dashboard statistics
- Debugging indexing issues
- Monitoring collection size

#### `index_into_vector_db(project, chunks, chunks_ids, do_reset, language)` → `bool`

Indexes document chunks into vector database with embeddings.

```
python
success = await nlp_controller.index_into_vector_db(
    project=project,
    chunks=chunk_list,              # List of DataChunk objects
    chunks_ids=[1, 2, 3, ...],      # Database IDs for chunks
    do_reset=False,                 # Keep existing vectors
    language=SupportedLanguages.ENGLISH
)
```

**Process Flow**:

1. **Generate Collection Name**: Based on project ID
2. **Extract Data**: Get texts and metadata from chunks
3. **Create Embeddings**: Convert texts to vectors using embedding client
4. **Create Collection**: Initialize vector DB collection (if not exists)
5. **Batch Insert**: Store vectors with metadata and IDs

**Parameters Explained**:

- `chunks`: List of `DataChunk` objects from database
- `chunks_ids`: Database record IDs for tracking
- `do_reset`: If `True`, clears existing vectors first
- `language`: For language-specific optimizations (tokenization, stemming)

**Embedding Process**:

```
python
vectors = self.embedding_client.embed_text(
    text=texts,
    document_type=DocumentTypeEnum.DOCUMENT.value  # Optimized for documents
)
```

**Why Batch Operations?**

- More efficient than inserting one-by-one
- Reduces network overhead
- Faster indexing for large document sets

#### `search_vector_db_collection(project, text, top_k)` → `list`

Performs semantic search and returns most relevant chunks.

```
python
results = await nlp_controller.search_vector_db_collection(
    project=project,
    text="What is the company's revenue?",
    top_k=5  # Return top 5 results
)
```

**Process Flow**:

1. **Generate Collection Name**: From project ID
2. **Embed Query**: Convert question to vector
3. **Vector Search**: Find similar chunks using cosine similarity
4. **Over-Retrieve**: Get `top_k * 10` results for reranking
5. **Rerank** (optional): Re-score results using LLM
6. **Return Top K**: Final results after reranking

**Why Over-Retrieve?**

- Vector similarity doesn't always capture semantic relevance
- Reranking with LLM improves result quality
- Trade-off: Retrieve 50, rerank to top 5

**Reranking Logic**:

```
python
if self.generation_client.rerank:
    reranked_docs = await self.generation_client.rerank(
        query=text,
        documents=results,
        top_n=top_k
    )
    return reranked_docs
```

**Return Format**:

```
python
[
    {
        "text": "Chunk content...",
        "metadata": {"page": 1, "source": "report.pdf"},
        "score": 0.87  # Relevance score
    },
    ...
]
```

#### `answer_rag_question(project, query, top_k)` → `(answer, prompt, history)`

Executes complete RAG pipeline: retrieve → augment → generate.

```
python
answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
    project=project,
    query="What are the key findings?",
    top_k=5
)
```

**Process Flow**:

**Step 1: Retrieval**

```
python
retrieved_documents = await self.search_vector_db_collection(
    project=project,
    text=query,
    top_k=top_k
)
```

**Step 2: Prompt Construction**

```
python
# System prompt (instructions for LLM)
system_prompt = "You are a helpful assistant that answers questions based on provided documents."

# Format retrieved documents
documents_prompts = "\n".join([
    f"Document {idx + 1}: {doc.text}"
    for idx, doc in enumerate(retrieved_documents)
])

# Add user query
footer_prompt = f"Question: {query}\nAnswer:"
```

**Step 3: Chat History**

```
python
chat_history = [
    {
        "role": "system",
        "content": system_prompt
    }
]
```

**Step 4: Generate Answer**

```
python
full_prompt = f"{documents_prompts}\n\n{footer_prompt}"
answer = await self.generation_client.generate_text(
    prompt=full_prompt,
    chat_history=chat_history
)
```

**Return Values**:

- `answer`: Generated response from LLM
- `full_prompt`: Complete prompt sent to LLM (for debugging)
- `chat_history`: Conversation context (for multi-turn dialogue)

**Template System**:
Uses `template_parser` for customizable prompts:

```
python
system_prompt = self.template_parser.get("rag", "system_prompt")
document_prompt = self.template_parser.get("rag", "document_prompt", {
    "doc_num": idx + 1,
    "chunk_text": doc.text
})
```

**Benefits**:

- Easy prompt customization without code changes
- Support for multiple prompt strategies
- Version control for prompts

---

### 6. EvaluationController

**Purpose**: Evaluates RAG pipeline quality using the Ragas framework.

**Location**: `controllers/EvaluationController.py`

**Responsibilities**:

- Run evaluation on test queries
- Generate answers and collect contexts
- Compute RAG metrics (faithfulness, relevancy, etc.)
- Produce evaluation reports

**Dependencies** (Injected):

- `nlp_controller`: For running RAG queries
- `ragas_provider`: Provides LLM and metrics for evaluation

**Key Concepts**:

- **Ragas**: RAG Assessment framework
- **Faithfulness**: Is answer grounded in retrieved context?
- **Answer Relevancy**: Does answer address the question?
- **Context Precision**: Quality of retrieved chunks
- **Context Recall**: Coverage of relevant information

**Key Methods**:

#### `run_evaluation_batch(project, test_queries)` → `pd.DataFrame`

Evaluates RAG system on a batch of test queries.

```
python
test_queries = [
    "What is the company's mission?",
    "Who is the CEO?",
    "What are the main products?"
]

results_df = await evaluation_controller.run_evaluation_batch(
    project=project,
    test_queries=test_queries
)
```

**Process Flow**:

**Step 1: Initialize Results Structure**

```
python
results_data = {
    "question": [],        # User queries
    "answer": [],          # Generated answers
    "contexts": [],        # Retrieved chunks
    "ground_truth": []     # Reference answers (optional)
}
```

**Step 2: Run RAG for Each Query**

```
python
for query in test_queries:
    # Retrieve relevant documents
    retrieved_docs = await self.nlp_controller.search_vector_db_collection(
        project=project,
        text=query
    )

    # Generate answer
    answer, _, _ = await self.nlp_controller.answer_rag_question(
        project=project,
        query=query
    )

    # Store results
    results_data["question"].append(query)
    results_data["answer"].append(answer)
    results_data["contexts"].append([doc.text for doc in retrieved_docs])
    results_data["ground_truth"].append("Reference answer if available")
```

**Step 3: Convert to Ragas Dataset**

```
python
dataset = Dataset.from_dict(results_data)
```

**Step 4: Run Evaluation**

```
python
result = await aevaluate(
    dataset=dataset,
    metrics=self.metrics,           # From ragas_provider
    llm=self.eval_llm,              # LLM for evaluation
    embeddings=self.eval_embeddings # Embeddings for evaluation
)
```

**Step 5: Return Results as DataFrame**

```
python
return result.to_pandas()
```

**Output Format**:

```
| question                  | answer              | faithfulness | answer_relevancy | context_precision |
|---------------------------|---------------------|--------------|------------------|-------------------|
| What is the mission?      | The mission is...   | 0.95         | 0.89             | 0.87              |
| Who is the CEO?           | The CEO is John...  | 1.00         | 0.92             | 0.95              |
```

**Metrics Explained**:

- **Faithfulness** (0-1): Answer is grounded in retrieved context
- **Answer Relevancy** (0-1): Answer addresses the question
- **Context Precision** (0-1): Retrieved chunks are relevant
- **Context Recall** (0-1): All relevant info was retrieved

**Ground Truth**: If you have reference answers, include them for more metrics:

```python
results_data["ground_truth"].append("The CEO is John Doe")
```

**Use Cases**:

- Benchmark different RAG configurations
- A/B test prompt strategies
- Monitor system performance over time
- Identify weak queries for improvement

---

## 🔄 Controller Interactions

### Typical RAG Query Flow

```
1. API Route receives query
        ↓
2. NLPController.answer_rag_question()
        ↓
3. NLPController.search_vector_db_collection()
   → Embedding client creates query vector
   → Vector DB returns similar chunks
   → (Optional) LLM reranks results
        ↓
4. NLPController constructs prompt
   → Template parser formats prompt
   → Includes retrieved chunks as context
        ↓
5. Generation client produces answer
        ↓
6. Return answer to user
```

### Document Upload Flow

```
1. API Route receives file upload
        ↓
2. DataController.validate_file()
   → Check file type and size
        ↓
3. DataController.generate_unique_file_path()
   → Create safe, unique file name
        ↓
4. Save file to disk (ProjectController path)
        ↓
5. ProcessController.get_file_content()
   → Parse document with appropriate loader
        ↓
6. ProcessController.process_file_content()
   → Split into chunks
        ↓
7. Store chunks in database (via ChunkModel)
        ↓
8. NLPController.index_into_vector_db()
   → Create embeddings
   → Store in vector database
```

---

## 🎯 Design Patterns

### 1. Dependency Injection

Controllers receive dependencies via constructor, not instantiation:

```
python
# Good: Testable and flexible
nlp_controller = NLPController(
    vectordb_client=pgvector_client,
    generation_client=gemini_client,
    embedding_client=gemini_embeddings,
    template_parser=parser
)

# Bad: Hard-coded dependencies
class NLPController:
    def __init__(self):
        self.vectordb_client = PGVectorClient()  # Can't swap providers
```

### 2. Provider Pattern

Controllers work with interfaces, not concrete implementations:

```
python
# Works with ANY vector DB that implements the interface
vectordb_client: VectorDBInterface = factory.create_vectordb()

# Controller doesn't care if it's pgvector, Pinecone, or Weaviate
await nlp_controller.index_into_vector_db(...)
```

### 3. Template Method Pattern

Base controller provides template, subclasses implement specifics:

```
python
class BaseController:
    def __init__(self):
        self.app_settings = get_settings()  # Common to all

class ProcessController(BaseController):
    def __init__(self, project_id: str):
        super().__init__()  # Get base functionality
        self.project_id = project_id  # Add specific state
```

---

## 🧪 Testing Controllers

### Unit Test Example

```
python
import pytest
from controllers import DataController
from fastapi import UploadFile

@pytest.mark.asyncio
async def test_file_validation():
    # Arrange
    data_controller = DataController()

    # Create mock upload file
    mock_file = UploadFile(
        filename="test.pdf",
        file=BytesIO(b"PDF content"),
        content_type="application/pdf"
    )
    mock_file.size = 1024 * 1024  # 1 MB

    # Act
    is_valid, message = data_controller.validate_file(mock_file)

    # Assert
    assert is_valid is True
    assert message == "FILE_VALIDATED_SUCCESS"
```

### Integration Test Example

```
python
@pytest.mark.asyncio
async def test_rag_pipeline(nlp_controller, test_project):
    # Arrange
    query = "What is the main topic?"

    # Act
    answer, prompt, history = await nlp_controller.answer_rag_question(
        project=test_project,
        query=query,
        top_k=5
    )

    # Assert
    assert answer is not None
    assert len(answer) > 0
    assert query in prompt
    assert len(history) > 0
```

---

## 📊 Performance Considerations

### Async Best Practices

**Do**: Use async for I/O operations

```
python
async def index_documents():
    vectors = embedding_client.embed_text(texts)  # I/O bound
    await vectordb_client.insert_many(vectors)    # I/O bound
```

**Don't**: Use async for CPU-bound operations

```
python
# Bad: Async doesn't help here
async def compute_statistics():
    return sum(large_list)  # CPU bound, blocks anyway
```

### Batch Operations

**Do**: Batch database operations

```python
# Good: One database call
await vectordb_client.insert_many(all_vectors)

# Bad: N database calls
for vector in all_vectors:
    await vectordb_client.insert_one(vector)
```

### Caching Strategies

Consider caching for:

- Frequently accessed prompts (template_parser)
- Project directory paths (ProjectController)
- Collection metadata (NLPController)

```
python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_project_files_dir(self, project_id: str) -> str:
    # Cached after first call
    return os.path.join(self.files_dir, project_id)
```

---

## ⚠️ Error Handling

Controllers should **raise exceptions** and let API routes handle them:

```
python
# In Controller
async def search_vector_db_collection(self, project, text, top_k):
    results = await self.vectordb_client.search_by_vector(...)

    if not results:
        raise ValueError("No results found")  # Let route handle it

    return results

# In Route
@app.post("/search")
async def search_endpoint(query: str):
    try:
        results = await nlp_controller.search_vector_db_collection(...)
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")
```

---

## 🔐 Security Considerations

### File Upload Security

**DataController** implements multiple security layers:

1. **Content Type Validation**: Whitelist approach
2. **File Size Limits**: Prevent DoS attacks
3. **File Name Sanitization**: Prevent path traversal
4. **Unique File Names**: Prevent overwrites

**Additional Recommendations**:

- Scan uploaded files for malware (ClamAV)
- Implement rate limiting on upload endpoints
- Store files outside web root
- Use separate storage service (S3, Azure Blob)

### Prompt Injection Prevention

**NLPController** should sanitize user
