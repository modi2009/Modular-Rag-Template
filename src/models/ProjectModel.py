from .BaseModel import BaseModel
from .db_schemas import Project
from .enums.DataBaseEnum import DataBaseEnum
from sqlalchemy.future import select
from sqlalchemy import func


class ProjectModel(BaseModel):
    def __init__(self, db_client: object):
        super().__init__(db_client)

    @classmethod
    async def create_instance(cls, db_client: object):
        return cls(db_client)
    async def create_project(self, project: Project) -> Project:
        async with self.db_client() as session:
            async with session.begin():
                session.add(project)
                await session.commit()
            await session.refresh(project)


        return project
                

    async def get_project_by_id(self, project_id: int) -> Project:
        async with self.db_client() as session:
            async with session.begin():
                query = select(Project).where(Project.project_id == int(project_id))
                result = await session.execute(query)
                project = result.scalar_one_or_none()
                if project:
                    return project
                else:
                    project = await self.create_project(Project(project_id= int(project_id)))
                    return project

    async def get_all_projects(self, page: int = 1, page_size: int = 10) -> list[Project]:
        async with self.db_client() as session:
            async with session.begin():
                query = select(func.count(Project.project_id))
                result = await session.execute(query)
                total_projects = result.scalar_one()

                total_pages = (total_projects + page_size - 1) // page_size
                if page > total_pages and total_pages != 0:
                    page = total_pages
                query = select(Project).offset((page - 1) * page_size).limit(page_size)
                result = await session.execute(query)
                projects = result.scalars().all()
                return projects