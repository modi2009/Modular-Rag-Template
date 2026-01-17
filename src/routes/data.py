from fastapi import APIRouter, UploadFile, Depends, status, Request
from fastapi.responses import JSONResponse
from logging import getLogger
from helpers import Settings, get_settings
from .schemas.data import ProcessRequest
from models import ResponseStatus, ProcessingEnum, ProjectModel, ChunkDataModel, AssetModel
from models.db_schemas.minirag import Project, DataChunk, Asset, RetrievedDocument
from models.enums.AssetEnum import AssetEnum
from controllers import DataController, ProjectController, ProcessController, NLPController
import aiofiles
import os

logger = getLogger('uvicorn.error')
data_router = APIRouter(
    prefix="/upload",
    tags=["upload"],
)


@data_router.post("/{project_id}")
async def upload_file(project_id: str, file: UploadFile, request : Request
                      , settings: Settings = Depends(get_settings)):
  


  #  insert project into database 
  project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
  project = await project_model.get_project_by_id(project_id=project_id)

  

  data_controller = DataController()
  is_valid, status_enum = data_controller.validate_file(file)

  if not is_valid:
      logger.error(f"File validation failed: {status_enum}")
      return JSONResponse(
          status_code=status.HTTP_400_BAD_REQUEST,
          content={"status": status_enum}
      )
  
  file_path, file_id = data_controller.generate_unique_file_path(
     project_id,
     file.filename
     
  )

  try:
     async with aiofiles.open(file_path, "wb") as f:
        while chunk := await file.read(settings.FILE_DEFAULT_CHUNK_SIZE):
           await f.write(chunk)
    
  except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseStatus.FILE_UPLOAD_FAILED.value
            }
        )
  
  asset = Asset(
      asset_type=AssetEnum.FILE.value,
      asset_name = file_id,
      asset_project_id = project.project_id,
      asset_size = os.path.getsize(file_path),
  )

  asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
  asset_record = await asset_model.create_asset(asset=asset)





  return JSONResponse(
            content={
                "signal": ResponseStatus.FILE_UPLOAD_SUCCESS.value,
                "file_id": asset_record.asset_name
            }
        )



@data_router.post("/process/{project_id}")
async def process_endpoint(request: Request, project_id: str, process_request: ProcessRequest):

    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)

    project = await project_model.get_project_by_id(project_id=project_id)

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)

    project_files_ids = {}

    print( process_request.file_id)
    print(type(project.project_id))

    if process_request.file_id: 
        asset = await asset_model.get_asset_record(
            asset_name=process_request.file_id,
            asset_project_id=project.project_id
        )

        if not asset:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseStatus.FILE_NOT_FOUND.value
                }
            )
        project_files_ids[asset.asset_id] = asset.asset_name

    else:
        assets = await asset_model.get_all_project_assets(
            asset_project_id=project.project_id,
            asset_type=AssetEnum.FILE.value
        )

        project_files_ids ={asset.asset_id : asset.asset_name for asset in assets}

    if len(project_files_ids) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseStatus.NO_FILES_TO_PROCESS.value
            }
        )
    process_controller = ProcessController(project_id=project_id)

    no_records = 0
    no_files = 0

    chunk_data_model = await ChunkDataModel.create_instance(db_client=request.app.db_client)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
        
     )

    if do_reset == 1:
        # delete associated vectors collection
        collection_name = nlp_controller.create_collection_name(project_id=project.project_id)
        _ = await request.app.vectordb_client.delete_collection(collection_name=collection_name)

        # delete associated chunks
        _ = await chunk_data_model.delete_chunks_by_project_id(
            project_id=project.project_id
        )

        print("Reset completed.")



    for asset_id, file_id in project_files_ids.items():

        file_content = process_controller.get_file_content(file_id=file_id)

        if file_content is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseStatus.FILE_READ_FAILED.value
                }
            )
        


        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )

        if file_chunks is None or len(file_chunks) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseStatus.PROCESSING_FAILED.value
                }
            )
        
        file_chunk_records = [
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_asset_id=asset_id,
                chunk_project_id=project.project_id,
                chunk_order = i + 1
            ) for i, chunk in enumerate(file_chunks)
        ]

        no_records = await chunk_data_model.insert_many_chunks(file_chunk_records)
        no_files += 1

    




    return JSONResponse(
        content={
            "signal": ResponseStatus.FILE_PROCESSING_COMPLETED.value,
            "files_processed": no_files,
            "records_created": no_records
        }
    )
  

    
    
