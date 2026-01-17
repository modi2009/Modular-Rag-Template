from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import (DistanceMethodEnums, PgVectorTableSchemeEnums, 
                             PgVectorDistanceMethodEnums, PgVectorIndexTypeEnums, SupportedLanguages)
import logging
from typing import List
from models.db_schemas import RetrievedDocument
from sqlalchemy.sql import text as sql_text
import json

# This class handles all interactions with PostgreSQL using the pgvector extension.
# It inherits from VectorDBInterface to ensure it has all required vector database methods.
class PGVectorProvider(VectorDBInterface):
    def __init__(self, db_client, distance_method: str,
                 default_vector_size: int,
                 index_threshold: int):
        """
        Initializes the provider with database connection settings and vector configurations.
        """
        self.db_client = db_client

        # Map general distance methods (Cosine, Dot Product) to specific PGVector SQL keywords
        if distance_method == DistanceMethodEnums.COSINE.value:
            distance_method = PgVectorDistanceMethodEnums.COSINE.value
        elif distance_method == DistanceMethodEnums.DOT.value:
            distance_method = PgVectorDistanceMethodEnums.DOT.value
        self.distance_method = distance_method

        self.default_vector_size = default_vector_size
        self.index_threshold = index_threshold  # Minimum records needed before creating an index
        self.logger = logging.getLogger("uvicorn.error")

        # Get the prefix for tables (e.g., 'items_') from enums to keep table names organized
        self.pgvector_prefix = PgVectorTableSchemeEnums._PREFIX.value

        # Helper to generate a consistent index name for a given table
        self.default_embed_index_name = lambda table_name: f"{self.pgvector_prefix}_{table_name}_vector_idx"
        self.default_gin_index_name = lambda table_name: f"{self.pgvector_prefix}_{table_name}_fts_idx"

    async def connect(self):       
        """
        Ensures the 'vector' extension is installed in PostgreSQL so it can handle embeddings.
        """
        async with self.db_client() as session:
            async with session.begin():
                # Runs the SQL command we discussed: teaching Postgres how to handle vectors
                await session.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await session.commit()
    
    async def disconnect(self):
        """Placeholder for closing connections if needed."""
        pass
    
    async def is_collection_existed(self, collection_name: str) -> bool:
        """
        Checks the PostgreSQL internal 'pg_tables' to see if a table already exists.
        """
        async with self.db_client() as session:
            async with session.begin():
              # Querying the system catalog for the table name
              query = sql_text(""" SELECT * FROM pg_tables WHERE tablename = :table_name """)
              result = await session.execute(query, {"table_name": collection_name})
              table = result.scalar_one_or_none()
              return True if table else False
    
    async def list_all_collections(self) -> List:
        """
        Returns a list of all tables that start with our specific pgvector prefix.
        """
        async with self.db_client() as session:
            async with session.begin():
              query = sql_text(""" SELECT tablename FROM pg_tables WHERE tablename like :prefix """)
              result = await session.execute(query, {"prefix": f"{self.pgvector_prefix}%"})
              tables = result.scalars().all()
              return tables
            
    async def get_collection_info(self, collection_name: str) -> dict:
        """
        Fetches metadata about a table (owner, space) and counts how many records are in it.
        """
        async with self.db_client() as session:
            async with session.begin():
                # Get system info about the table
                table_inf_query = sql_text(""" 
                    SELECT schemaname, tablename, tableowner, tablespace, hasindexes 
                    FROM pg_tables WHERE tablename = :table_name
                                                                        """)
                # Count total rows
                table_count_query = sql_text(f"SELECT COUNT(*) FROM {collection_name}")
                
                table_info_result = await session.execute(table_inf_query, {"table_name": collection_name})
                table_info = table_info_result.fetchone()


                
                count_result = await session.execute(table_count_query)
                record_count = count_result.scalar_one()

                if not table_info:
                    return None
                print(table_info)

                return {
                    "schemaname": table_info.schemaname,
                    "tablename": table_info.tablename,
                    "tableowner": table_info.tableowner,
                    "tablespace": table_info.tablespace,
                    "hasindexes": table_info.hasindexes,
                    "record_count": record_count
                }

    async def delete_collection(self, collection_name: str)-> bool:
        """
        Permanently deletes a table (collection) from the database.
        """
        async with self.db_client() as session:
            async with session.begin():
                self.logger.info(f"Dropping table {collection_name}...")
                drop_query = sql_text(f"DROP TABLE IF EXISTS {collection_name}")
                await session.execute(drop_query)
                await session.commit()

        return True

    async def create_collection(self, collection_name: str, 
                                embedding_size: int,
                                do_reset: bool = False)-> None:
        """
        Creates a new table structured for RAG: ID, Text, Vector, and Metadata.
        """
        async with self.db_client() as session:
            async with session.begin():
                # If do_reset is True, delete the old table first
                if do_reset:
                    _ = await self.delete_collection(collection_name=collection_name)
                    
                
                # 1. Create the table without the GENERATED ALWAYS column
                create_table_sql = sql_text(f"""
                    CREATE TABLE IF NOT EXISTS {collection_name} (
                        id SERIAL PRIMARY KEY,
                        {PgVectorTableSchemeEnums.TEXT.value} TEXT,
                        {PgVectorTableSchemeEnums.VECTOR.value} VECTOR({embedding_size}),
                        {PgVectorTableSchemeEnums.CHUNK_ID.value} INTEGER,
                        {PgVectorTableSchemeEnums.LANGUAGE.value} TEXT DEFAULT 'english',
                        {PgVectorTableSchemeEnums.FTS_TOKENS.value} TSVECTOR, -- No longer generated
                        {PgVectorTableSchemeEnums.METADATA.value} JSONB DEFAULT '{{}}'
                    );
                """)

                # 2. Create a function to handle the multi-language tokenization
                create_function_sql = sql_text(f"""
                    CREATE OR REPLACE FUNCTION {collection_name}_tsvector_trigger() RETURNS trigger AS $$
                    BEGIN
                    NEW.{PgVectorTableSchemeEnums.FTS_TOKENS.value} := 
                        to_tsvector(NEW.{PgVectorTableSchemeEnums.LANGUAGE.value}::regconfig, NEW.{PgVectorTableSchemeEnums.TEXT.value});
                    RETURN NEW;
                    END
                    $$ LANGUAGE plpgsql;
                """)

                # 3. Attach the trigger to the table
                create_trigger_sql = sql_text(f"""
                    CREATE OR REPLACE TRIGGER {collection_name}_tsvector_update
                    BEFORE INSERT OR UPDATE ON {collection_name}
                    FOR EACH ROW EXECUTE FUNCTION {collection_name}_tsvector_trigger();
                """)

                # Execute these in order within your session
                await session.execute(create_table_sql)
                await session.execute(create_function_sql)
                await session.execute(create_trigger_sql)
                await session.commit()
                
                # If we have enough data, create an HNSW index to make searches faster
                if embedding_size >= self.index_threshold:
                    await self.create_all_indexes(collection_name)

    async def is_index_existed(self, collection_name: str, indexing_method) -> bool:
        """
        Checks the 'pg_indexes' system table to see if our vector index is already built.
        """
        if indexing_method == "embed":
            index_name = self.default_embed_index_name(collection_name)
        elif indexing_method == "fts":
            index_name = self.default_gin_index_name(collection_name)

        async with self.db_client() as session:
            async with session.begin():
                check_sql = sql_text(""" 
                                    SELECT 1 FROM pg_indexes 
                                    WHERE tablename = :collection_name AND indexname = :index_name
                                    """)

                results = await session.execute(check_sql, {"index_name": index_name, "collection_name": collection_name})
                return bool(results.scalar_one_or_none())
            
    async def _create_embed_vector_index(self, collection_name: str,
                                        index_type: str = PgVectorIndexTypeEnums.HNSW.value)-> None:
        """
        Builds a high-speed search index (HNSW) on the vector column.
        """
        if await self.is_index_existed(collection_name, indexing_method="embed"):
            return False
        
        async with self.db_client() as session:
            async with session.begin():
                
                # Create the index using the chosen distance method (Cosine/Dot)
                index_name = self.default_index_name(collection_name, index_type)
                create_idx_sql = sql_text(
                    f'CREATE INDEX {index_name} ON {collection_name} '
                    f'USING {index_type} ({PgVectorTableSchemeEnums.VECTOR.value} {self.distance_method})'
                )
                await session.execute(create_idx_sql)

    async def _create_gin_vector_index(self, collection_name: str)-> None:
        """Builds a GIN index for the keyword/text search column."""
        if await self.is_index_existed(collection_name, indexing_method="fts"):
            return False
        
        async with self.db_client() as session:
            async with session.begin():
                
                # Create the index using the chosen distance method (Cosine/Dot)
                index_name = self.default_gin_index_name(collection_name)
                create_idx_sql = sql_text(f"""
                    CREATE INDEX {index_name} ON {collection_name} 
                    USING GIN ({PgVectorTableSchemeEnums.FTS_TOKENS.value})
                    """
                )
                await session.execute(create_idx_sql)

    async def create_all_indexes(self, collection_name: str, 
                                  index_type: str = PgVectorIndexTypeEnums.HNSW.value) -> bool:
        """Checks record threshold and builds both Vector and Keyword indexes."""
        async with self.db_client() as session:
            async with session.begin():
                res = await session.execute(sql_text(f"SELECT COUNT(*) FROM {collection_name}"))
                count = res.scalar_one()

                if count < self.index_threshold:
                    self.logger.info(f"Not enough records ({count}) to create index on {collection_name}. Threshold is {self.index_threshold}.")
                    return False
                else:
                    self.logger.info(f"Creating index on {collection_name} with {count} records.")
                    await self._create_embed_vector_index(collection_name, index_type)
                    await self._create_gin_vector_index(collection_name, index_type)
                    return True

    async def reset_vector_index(self, collection_name: str, 
                                       index_type: str = PgVectorIndexTypeEnums.HNSW.value) -> bool:
        """Deletes and recreates the index (useful if data changed significantly)."""
        index_embed_name = self.default_index_name(collection_name)
        index_gin_name = self.default_gin_index_name(collection_name)
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text(f'DROP INDEX IF EXISTS {index_embed_name}'))
                await session.execute(sql_text(f'DROP INDEX IF EXISTS {index_gin_name}'))
        return await self.create_all_indexes(collection_name, index_type)

    async def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, record_id: str = None, language: SupportedLanguages = SupportedLanguages.ENGLISH, index_type: str = PgVectorIndexTypeEnums.HNSW.value) -> None:
        """Inserts a single document and its embedding into the table."""
        async with self.db_client() as session:
            async with session.begin():
                
                insert_query = sql_text(f"""
                    INSERT INTO {collection_name} 
                    ({PgVectorTableSchemeEnums.TEXT.value}, {PgVectorTableSchemeEnums.VECTOR.value}, 
                     {PgVectorTableSchemeEnums.CHUNK_ID.value}, {PgVectorTableSchemeEnums.METADATA.value}, {language.value})
                    VALUES (:text, :vector, :chunk_id, :metadata, :language);
                """)
                await session.execute(insert_query, {
                    "text": text, "vector": vector, "chunk_id": record_id,
                    "metadata": json.dumps(metadata) if metadata else None
                })
                await session.commit()
        # Ensure index exists/updates after insertion
        await self.create_all_indexes(collection_name, index_type)

    async def insert_many(self, collection_name: str, texts: list,
                         vectors: list, metadata: list = None,
                         record_ids: list = None, batch_size: int = 50, index_type: str = PgVectorIndexTypeEnums.HNSW.value, language = SupportedLanguages.ENGLISH.value) -> bool:
        
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error(f"Can not insert new records to non-existed collection: {collection_name}")
            return False
        
        if len(vectors) != len(record_ids):
            self.logger.error(f"Invalid data items for collection: {collection_name}")
            return False
        
        if not metadata or len(metadata) == 0:
            metadata = [None] * len(texts)
        
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_vectors = vectors[i:i + batch_size]
                    batch_metadata = metadata[i:i + batch_size]
                    batch_record_ids = record_ids[i:i + batch_size]

                    values = []

                    for _text, _vector, _metadata, _record_id in zip(batch_texts, batch_vectors, batch_metadata, batch_record_ids):
                        
                        metadata_json = json.dumps(_metadata, ensure_ascii=False) if _metadata is not None else "{}"
                        values.append({
                            'text': _text,
                            'vector': "[" + ",".join([ str(v) for v in _vector ]) + "]",
                            'metadata': metadata_json,
                            'chunk_id': _record_id,
                            'language': language.value,
                        })
                    
                    batch_insert_sql = sql_text(f'INSERT INTO {collection_name} '
                                    f'({PgVectorTableSchemeEnums.TEXT.value}, '
                                    f'{PgVectorTableSchemeEnums.VECTOR.value}, '
                                    f'{PgVectorTableSchemeEnums.METADATA.value}, '
                                    f'{PgVectorTableSchemeEnums.CHUNK_ID.value}, '
                                    f'{PgVectorTableSchemeEnums.LANGUAGE.value}) '
                                    f'VALUES (:text, :vector, :metadata, :chunk_id, :language)')
                    
                    await session.execute(batch_insert_sql, values)

        await self.create_all_indexes(collection_name=collection_name, index_type=index_type)

        return True
    

    async def search_by_vector(self, collection_name: str, query_text: str, vector: list, 
                                top_k: int, rrf_k: int = 60)-> List[RetrievedDocument]:
            """
            Modified RAG function: Combines Vector and Keyword search using RRF.
            """
            if not await self.is_collection_existed(collection_name):
                print(f"Collection {collection_name} does not exist.")
                return False
            
            # Format vector for Postgres
            vector_str = "[" + ",".join([str(v) for v in vector]) + "]"
            
            async with self.db_client() as session:
                async with session.begin():
                    # We use a CTE (Common Table Expression) to rank results from both 'brains'
                    search_sql = sql_text(f"""
                        WITH vector_results AS (
                            SELECT {PgVectorTableSchemeEnums.ID.value}, 
                                ROW_NUMBER() OVER (ORDER BY {PgVectorTableSchemeEnums.VECTOR.value} <=> :vector) as rank
                            FROM {collection_name}
                            LIMIT :top_k
                        ),
                        keyword_results AS (
                            SELECT {PgVectorTableSchemeEnums.ID.value}, 
                                ROW_NUMBER() OVER (ORDER BY ts_rank_cd({PgVectorTableSchemeEnums.FTS_TOKENS.value}, plainto_tsquery(:query)) DESC) as rank
                            FROM {collection_name}
                            WHERE {PgVectorTableSchemeEnums.FTS_TOKENS.value} @@ plainto_tsquery(:query)
                            LIMIT :top_k
                        )
                        SELECT 
                            t.{PgVectorTableSchemeEnums.TEXT.value} as text,
                            (COALESCE(1.0 / (:rrf_k + v.rank), 0.0) + 
                            COALESCE(1.0 / (:rrf_k + k.rank), 0.0)) as score
                        FROM vector_results v
                        FULL OUTER JOIN keyword_results k ON v.id = k.id
                        JOIN {collection_name} t ON t.id = COALESCE(v.id, k.id)
                        ORDER BY score DESC
                        LIMIT :top_k
                    """)
                    
                    result = await session.execute(search_sql, {
                        "vector": vector_str, 
                        "query": query_text, 
                        "top_k": top_k,
                        "rrf_k": rrf_k
                    })
                    records = result.fetchall()

                    return [
                        RetrievedDocument(text=record.text, score=record.score)
                        for record in records
                    ]