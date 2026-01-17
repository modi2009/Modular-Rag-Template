from .BaseModel import BaseModel
from .db_schemas import DataChunk
from .enums.DataBaseEnum import DataBaseEnum
from sqlalchemy.future import select
from sqlalchemy import func, delete
from bson.objectid import ObjectId

# ChunkDataModel manages the actual "pieces" of text (chunks) extracted from documents.
# In a RAG system, documents are split into these chunks to be converted into vectors.
class ChunkDataModel(BaseModel):
    def __init__(self, db_client: object):
        # Initializes the parent BaseModel with the database client
        super().__init__(db_client)

    @classmethod
    async def create_instance(cls, db_client: object):
        """Helper factory method to create an instance of this model."""
        return cls(db_client)
    
    async def create_chunk(self, chunk: DataChunk) -> DataChunk:
        """Saves a single new text chunk to the database."""
        async with self.db_client() as session:
            async with session.begin():
                # Add the chunk object to the session (workspace)
                session.add(chunk)
                # Permanently save it to the database
                await session.commit()
                # Refresh the object to get any database-generated values (like auto-increment IDs)
                await session.refresh(chunk)
        return chunk
    
    async def get_chunk_by_id(self, chunk_id: int) -> DataChunk :
        """Finds one specific chunk using its primary key ID."""
        async with self.db_client() as session:
            async with session.begin():
                # Create a SELECT query for a specific ID
                query = select(DataChunk).where(DataChunk.chunk_id == chunk_id)
                result = await session.execute(query)
                # Use scalar_one_or_none to safely get 1 object or None if not found
                chunk = result.scalar_one_or_none()
                return chunk
            
    async def insert_many_chunks(self, chunks: list, batch_size: int=100):
        """
        Saves a large list of chunks efficiently using batching.
        This is much faster than calling create_chunk() 100 times.
        """
        async with self.db_client() as session:
            async with session.begin():
                # Loop through the list in steps of 'batch_size'
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    # add_all prepares multiple objects for the database at once
                    session.add_all(batch)
            # Commit the entire batch group
            await session.commit()
        return len(chunks)

    async def delete_chunks_by_project_id(self, project_id: ObjectId):
        """
        Deletes all chunks associated with a specific project.
        Useful when a user deletes a project or re-uploads a document.
        """
        async with self.db_client() as session:
            # Create a DELETE statement instead of a SELECT
            stmt = delete(DataChunk).where(DataChunk.chunk_project_id == project_id)
            result = await session.execute(stmt)
            await session.commit()
            # Returns the number of rows actually deleted
            return result.rowcount
    
    async def get_poject_chunks(self, project_id: ObjectId, page_no: int=1, page_size: int=50):
        """
        Retrieves chunks for a specific project using Pagination.
        Defaults to 50 chunks per page to keep response sizes manageable.
        """
        async with self.db_client() as session:
            # offset skips previous pages, limit restricts results to the current page size
            stmt = select(DataChunk).where(DataChunk.chunk_project_id == project_id).offset((page_no - 1) * page_size).limit(page_size)
            result = await session.execute(stmt)
            # Flatten the result into a clean Python list of objects
            records = result.scalars().all()
        return records
    
    async def get_total_chunks_count(self, project_id: ObjectId):
        """
        Counts how many total chunks belong to a project.
        Used by the UI to calculate how many pages of data exist.
        """
        total_count = 0
        async with self.db_client() as session:
            # Uses the SQL COUNT() function on the primary key for maximum speed
            count_sql = select(func.count(DataChunk.chunk_id)).where(DataChunk.chunk_project_id == project_id)
            records_count = await session.execute(count_sql)
            # scalar() extracts the single number from the result row
            total_count = records_count.scalar()
        
        return total_count