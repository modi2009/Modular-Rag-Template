from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    APP_NAME: str
    APP_VERSION: str
    GEMINI_API_KEY: str
    FILE_ALLOWED_TYPES: list[str]
    FILE_MAX_SIZE: int  # in MB
    FILE_DEFAULT_CHUNK_SIZE: int  # in KB

    GENERATION_BACKEND: str
    GENERATION_MODEL_ID: str
    EMBEDDING_BACKEND: str
    EMBEDDING_MODEL_ID: str
    RAGAS_PROVIDER: str
    EMBEDDING_MODEL_SIZE: int
    INPUT_DAFAULT_MAX_CHARACTERS: int
    GENERATION_DAFAULT_MAX_TOKENS: int
    GENERATION_DAFAULT_TEMPERATURE: float
    SYSTEM_INSTRUCTIONS: str

    VECTOR_DB_PATH: str
    VECTOR_DB_BACKEND: str
    VECTOR_DB_DISTANCE_METHOD: str
    VECTOR_DB_PGVEC_INDEX_THRESHOLD: int
    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str

    PRIMARY_LANG: str
    DEFAULT_LANG: str


def get_settings():
    return Settings()
    
