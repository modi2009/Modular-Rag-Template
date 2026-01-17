from fastapi import FastAPI
from routes.base import base_router
from routes.data import data_router
from routes.nlp import nlp_router
from routes.evaluation import eval_router
import os
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.Ragas.RAGASLLMBuilder import RagasFactory
from stores.llm.templates.template_parser import TemplateParser
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Initialize the FastAPI application
app = FastAPI()

async def startup_span():
    """
    This function runs once when the server starts. 
    It initializes all heavy connections (DB, LLM, VectorDB).
    """
    # 1. Load environment variables and settings (DB credentials, API keys, etc.)
    settings = get_settings()

    # 2. Build the Async Connection String for PostgreSQL
    # Note the use of 'postgresql+asyncpg' which is required for async SQLAlchemy
    postgres_conn = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"

    # 3. Create the Engine (The "Physical Connection" pool)
    app.db_engine = create_async_engine(postgres_conn)

    # 4. Create the SessionMaker (The "Workspace Factory")
    # expire_on_commit=False prevents SQLAlchemy from "forgetting" data after a commit
    app.db_client = sessionmaker(
        app.db_engine, class_=AsyncSession, expire_on_commit=False
    )

    # 5. Initialize Factories for LLMs and Vector Databases
    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(config=settings, db_client=app.db_client)
    raga_factory = RagasFactory(config=settings)
    app.ragas_provider = raga_factory.get_provider(provider_type=settings.RAGAS_PROVIDER)
    app.ragas_provider.get_llm(model_id=settings.GENERATION_MODEL_ID, system_instructions=settings.SYSTEM_INSTRUCTIONS)
    app.ragas_provider.get_embeddings(model_id=settings.EMBEDDING_MODEL_ID)

    # 6. Setup Generation Client (e.g., OpenAI/Anthropic for talking)
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID, system_instructions=settings.SYSTEM_INSTRUCTIONS)

    # 7. Setup Embedding Client (Converts text into lists of numbers/vectors)
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)
    
    # 8. Setup Vector DB Client (Postgres with pgvector or Qdrant)
    app.vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    # Execute the 'CREATE EXTENSION' command we discussed earlier
    await app.vectordb_client.connect()

    # 9. Setup Template Parser (Manages prompts in different languages)
    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )



async def shutdown_span():
    """
    This function runs when the server stops. 
    It cleans up connections so the database doesn't stay 'locked'.
    """
    app.db_engine.dispose()
    await app.vectordb_client.disconnect()

# Register the startup and shutdown handlers
app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

# Register the API routes (the URLs you will call from your frontend)
app.include_router(base_router)
app.include_router(data_router)
app.include_router(nlp_router)
app.include_router(eval_router)
