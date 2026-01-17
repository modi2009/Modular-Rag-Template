from .BaseModel import BaseModel
from .db_schemas import Asset
from .enums.DataBaseEnum import DataBaseEnum
from bson import ObjectId
from sqlalchemy.future import select

# AssetModel manages the metadata for files uploaded to a project.
# It tracks things like the file name, its type (PDF/Text), and which project it belongs to.
class AssetModel(BaseModel):

    def __init__(self, db_client: object):
        # Initialize the base class and store the database client
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        """Factory method to create a new instance of AssetModel."""
        instance = cls(db_client)
        return instance

    async def create_asset(self, asset: Asset):
        """
        Takes an Asset object (a new file record) and saves it to the database.
        """
        # Open a new asynchronous session
        async with self.db_client() as session:
            # Start a database transaction
            async with session.begin():
                # Add the asset metadata to the database session
                session.add(asset)
            # Commit (save permanently) and refresh to get any DB-generated fields
            await session.commit()
            await session.refresh(asset)
        return asset

    async def get_all_project_assets(self, asset_project_id: str, asset_type: str):
        """
        Retrieves all assets for a specific project that match a specific type.
        Example: Find all 'PDF' assets for Project X.
        """
        async with self.db_client() as session:
            # Select assets where both the project ID and the file type match
            stmt = select(Asset).where(
                Asset.asset_project_id == asset_project_id,
                Asset.asset_type == asset_type
            )
            result = await session.execute(stmt)
            # .scalars().all() flattens the result into a clean list of Asset objects
            records = result.scalars().all()
        return records

    async def get_asset_record(self, asset_project_id: str, asset_name: str):
        """
        Finds a specific file record by its name within a project.
        Useful for checking if a file with the same name has already been uploaded.
        """
        async with self.db_client() as session:
            # Filter by project ID and the specific filename
            stmt = select(Asset).where(
                Asset.asset_project_id == asset_project_id,
                Asset.asset_name == asset_name
            )
            result = await session.execute(stmt)
            # Returns the Asset object if found, otherwise returns None
            record = result.scalar_one_or_none()
        return record