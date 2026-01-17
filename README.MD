# 🚀 RAG Evaluation System with FastAPI

A production-ready, modular RAG (Retrieval-Augmented Generation) pipeline built with FastAPI, featuring an automated evaluation suite powered by Ragas and Google Gemini. Designed with a **provider pattern architecture** for maximum flexibility—easily swap LLMs, vector databases, and evaluation frameworks without touching core logic.

---

## 🌟 Key Features

- **Provider-Based Architecture**: Generic interfaces with factory patterns for LLMs, vector databases, and evaluation frameworks
- **Multi-Format Support**: Process PDF, TXT, and other document formats seamlessly
- **Intelligent Chunking**: Optimized document chunking with metadata preservation
- **Vector Search**: Powered by configurable vector database providers (e.g., pgvector)
- **RAG Evaluation**: Built-in evaluation pipeline using Ragas framework
- **RESTful API**: Clean, documented endpoints for data ingestion, querying, and evaluation
- **Environment-Based Configuration**: Complete control via `.env` files
- **Docker Ready**: Containerized deployment for consistency across environments

---

## 📁 Project Structure

```
rag-project/
├── docker/                      # Docker configuration files
├── src/
│   ├── models/                  # Core data models
│   │   ├── BaseModel.py        # Base model with common functionality
│   │   ├── AssetModel.py       # Asset (document) management model
│   │   ├── ProjectModel.py     # Project (row) management model
│   │   ├── ChunkDataModel.py       # Document chunk model
│   │   ├── enums/              # Enumeration definitions
│   │   │   ├── ResponseEnums.py       # API response types
│   │   │   ├── ProcessEnums.py        # Asset types (PDF, TXT, etc.)
│   │   │   ├── DataBaseEnums.py       # DB naming conventions
│   │   │   └── AssetEnums.py          # Asset-specific enumerations
│   │   └── dbSchemas/          # Database schemas
│   │       ├── asset.py
│   │       ├── project.py
│   │       └── datachunk.py
│   ├── controllers/             # Business logic controllers
│   │   ├── BaseController.py           # Base controller functionality
│   │   ├── ProjectController.py        # get project directory
│   │   ├── ProcessController.py        # Document processing logic and chuking
│   │   ├── DataController.py           # Data management data validation and create unique name
│   │   ├── NLPController.py            # RAG query handling
│   │   └── EvaluationController.py     # RAG evaluation logic
│   ├── routes/                  # API route definitions
│   │   ├── base.py             # Base routes
│   │   ├── data.py             # Data processing & storage endpoints
│   │   ├── nlp.py              # RAG query endpoints
│   │   └── evaluation.py       # Evaluation endpoints
│   ├── stores/                  # Provider implementations
│   │   ├── LLM/                # LLM provider abstraction
│   │   │   ├── LLMInterface.py
│   │   │   ├── LLMProviderFactory.py
│   │   │   ├── LLMEnums.py
│   │   │   └── providers/      # Specific LLM implementations (e.g., Gemini)
│   │   ├── vectorDB/           # Vector database abstraction
│   │   │   ├── VectorDBInterface.py
│   │   │   ├── VectorDBEnums.py
│   │   │   ├── VectorDBProviderFactory.py
│   │   │   └── providers/      # Specific DB implementations (e.g., pgvector)
│   │   └── ragas/              # Ragas evaluation abstraction
│   │       ├── RAGASLLMBuilder.py
│   │       ├── RAGASLLMInterface.py
│   │       └── providers/      # Ragas implementations with different LLMs
│   ├── assets/                  # Document storage directory
│   └── helpers/                 # Utility functions
│       └── config.py           # Configuration management
├── .env                         # Environment variables
├── requirements.txt             # Python dependencies
├── main.py                      # FastAPI application entrypoint
└── README.md                    # This file
```

---

## 🏗️ Architecture Overview

### Provider Pattern Design

This project implements a **provider pattern** to decouple core logic from specific implementations:

- **Interfaces**: Define contracts for LLMs, vector databases, and evaluation tools
- **Factories**: Dynamically instantiate providers based on environment configuration
- **Base Classes**: Provide common functionality across implementations
- **Providers**: Concrete implementations (e.g., Gemini LLM, pgvector DB)

**Benefits:**

- Switch from Gemini to OpenAI by changing one environment variable
- Swap pgvector for Pinecone without modifying application code
- Test with different evaluation frameworks seamlessly

### Data Flow

1. **Ingestion**: Upload documents → `DataController` → Asset storage
2. **Processing**: Parse documents → Chunk → Embed → Store in vector DB
3. **Query**: User question → `NLPController` → Vector search → LLM generation
4. **Evaluation**: Query logs → `EvaluationController` → Ragas metrics

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **WSL2** (Ubuntu 20.04+ recommended) or Linux environment
- **Python 3.10+**
- **PostgreSQL** with **pgvector extension** installed
- **Docker** & **Docker Compose** (optional, for containerized deployment)
- **Google Gemini API Key** (obtain from [Google AI Studio](https://makersuite.google.com/app/apikey))

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

Open your WSL/Linux terminal:

```bash
git clone
cd rag-project
```

### 2. Create a Virtual Environment

Using a virtual environment prevents dependency conflicts:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

Upgrade pip and install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following structure:

```env
APP_NAME="Your App Name"
APP_VERSION="Your App Version"
FILE_ALLOWED_TYPES= ["text/plain","document/pdf", "document/txt", "document/docx","application/pdf"]
FILE_MAX_SIZE=100
FILE_DEFAULT_CHUNK_SIZE=512 # 512KB




POSTGRES_USERNAME="Postgress Username"
POSTGRES_PASSWORD="Postgress Password"
POSTGRES_HOST="Host"
POSTGRES_PORT= # port number integer
POSTGRES_MAIN_DATABASE="DATABASE Name"

# ========================= LLM Config =========================
# --- LLM Provider Settings ---
GENERATION_BACKEND="Model Provider"
EMBEDDING_BACKEND="Model provider"
GEMINI_API_KEY="API key"
RAGAS_PROVIDER = "Model Provider"

# --- Model Identifiers ---
GENERATION_MODEL_ID="Model ID"
EMBEDDING_MODEL_ID="Model ID"
EMBEDDING_MODEL_SIZE= # Embedding dimension

# --- Default Parameters ---
INPUT_DAFAULT_MAX_CHARACTERS=1024
GENERATION_DAFAULT_MAX_TOKENS=200
GENERATION_DAFAULT_TEMPERATURE=0.1
SYSTEM_INSTRUCTIONS="You are a helpful assistant that provides accurate and concise answers based on the provided context."

# ========================= Vector DB Config =========================
VECTOR_DB_BACKEND = "DB Provider"
VECTOR_DB_PATH = "DB path"
VECTOR_DB_DISTANCE_METHOD = "Distance Method"
VECTOR_DB_PGVEC_INDEX_THRESHOLD = 1000

# ========================= Template Configs =========================
PRIMARY_LANG = "en"
DEFAULT_LANG = "en"

```

**Security Note**: Never commit `.env` files to version control. Add `.env` to your `.gitignore`.

### 5. Set Up PostgreSQL with pgvector inside docker

## 🚀 Running the Application

### Development Mode (with hot-reload)

FastAPI uses `uvicorn` with `uvloop` for high performance in WSL:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Base URL**: http://localhost:8000

### Production Mode

For production deployment without auto-reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

Use Docker Compose for containerized deployment:

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 📡 API Endpoints

### Data Management (`/data`)

- **POST** `/data/upload/{project_id}` - Upload documents (PDF, TXT, etc.)
- **POST** `/data/upload/process/{project_id}` - process data and insert it into database
  parameters:
  file_id: str
  chunk_size: Optional[int] = 100
  overlap_size: Optional[int] = 20
  do_reset: Optional[int] = 0

### RAG Queries (`/nlp`)

- **POST** `/nlp/push/{project_id}` push vectorized data into vector db
  parameters
  do_reset: Optional[int] = 0
- **GET** `/nlp/info/{project_id}` get info of vectorized data from db
- **POST** `/nlp/search/{project_id}` search vector db using hybrid search and reranking
  parameters:
  text: str
  top_k: Optional[int] = 5
- **POST** `/nlp/answer/{project_id}` answer rag question
  text: str
  top_k: Optional[int] = 5

### Evaluation (`/evaluation`)

- **POST** `/evaluation/{project_id}` - Run Ragas evaluation on query results
  parameters:
  test_queries: List[str]

---

## 🔧 Configuration & Customization

### Switching LLM Providers

To switch from Gemini to OpenAI, update your `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4
```

The factory pattern automatically loads the correct provider—no code changes needed.

### Changing Vector Databases

To use Pinecone instead of pgvector:

```env
VECTOR_DB_PROVIDER=pinecone
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX=rag-index
```

### Adjusting Chunking Strategy

Modify chunking parameters in `.env`:

```env
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

Or extend `ChunkModel.py` to implement custom chunking logic.

---

## 🧪 Testing & Evaluation

**Metrics Provided:**

- **Answer Relevancy**: How relevant is the answer to the question?
- **Faithfulness**: Is the answer grounded in retrieved context?
- **Context Precision**: Quality of retrieved chunks
- **Context Recall**: Coverage of relevant information

For production, integrate with tools like:

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Prometheus** + **Grafana**
- **DataDog** or **New Relic**

---

## 🐳 Docker Configuration

The `docker/` folder contains:

- **Dockerfile**: Application container definition
- **docker-compose.yml**: Multi-service orchestration (FastAPI + PostgreSQL)
- **.dockerignore**: Exclude unnecessary files from builds

### Environment Variables in Docker

Create a `docker/.env` file for Docker-specific configurations:

```env
POSTGRES_PASSWORD=secure_password
GEMINI_API_KEY=your_key
```

Reference in `docker-compose.yml`:

```yaml
env_file:
  - .env
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

# big thanks to Engineer Abu Bakr Soliman this repo is built based on mini rag series in YouTube

## 🆘 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'fastapi'`  
**Solution**: Ensure virtual environment is activated and dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Issue**: PostgreSQL connection refused  
**Solution**: Check PostgreSQL is running:

```bash
sudo service postgresql status
sudo service postgresql start
```

**Issue**: pgvector extension not found  
**Solution**: Install pgvector for your PostgreSQL version:

```bash
sudo apt install postgresql-14-pgvector
```

---

**Built with ❤️ using FastAPI, Gemini, and Ragas**
