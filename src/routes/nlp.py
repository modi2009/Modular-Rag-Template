from fastapi import FastAPI, APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemas.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkDataModel import ChunkDataModel
from controllers import NLPController
from models import ResponseStatus
from tqdm.auto import tqdm

import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/nlp",
    tags=["nlp"],
)


@nlp_router.post("/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):

  print(project_id)
  print("haha")
  project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
  project = await project_model.get_project_by_id(project_id=int(project_id))
  if not project:
      return JSONResponse(
          status_code=status.HTTP_404_NOT_FOUND,
          content={"signal": ResponseStatus.PROJECT_NOT_FOUND.value}
      )
  
  chunk_data_model = await ChunkDataModel.create_instance(db_client=request.app.db_client)

  nlp_controller = NLPController(
      vectordb_client=request.app.vectordb_client,
      generation_client=request.app.generation_client,
      embedding_client=request.app.embedding_client,
      template_parser=request.app.template_parser
  )

  has_records = True
  page_no = 1
  inserted_count = 0
  idx = 0

  collection_name = nlp_controller.create_collection_name(project_id=project.project_id)

  total_chunks_count = await chunk_data_model.get_total_chunks_count (project_id=project.project_id)

  _ = await request.app.vectordb_client.create_collection(
      collection_name=collection_name,
      embedding_size=request.app.embedding_client.embedding_size,
      do_reset=push_request.do_reset)
  pbar = tqdm(total=total_chunks_count, desc="Indexing Chunks", unit="chunks")

  while has_records:
      chunks = await chunk_data_model.get_poject_chunks(
          project_id=project.project_id,
          page_no=page_no,
      )
      if len(chunks):
          page_no += 1
      if not chunks:
          has_records = False
          break
      
      chunk_ids = [ c.chunk_id for c in chunks ]
      inserted_count += 1
      idx += len(chunks)

      is_inserted = await nlp_controller.index_into_vector_db(
          project=project,
          chunks=chunks,
          chunks_ids=chunk_ids,
      )

      if not is_inserted:
          logger.error(f"Failed to index chunks for project {project.project_id} at page {page_no}.")
          return JSONResponse(
              status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
              content={"signal": ResponseStatus.INDEXING_FAILED.value}
          )
      
      pbar.update(len(chunks))
      inserted_count += len(chunks)

  return JSONResponse(
      status_code=status.HTTP_200_OK,
      content={
          "signal": ResponseStatus.INDEXING_COMPLETED.value,
          "indexed_chunks": inserted_count
      })

@nlp_router.get("/collection_info/{project_id}")
async def get_collection_info(request: Request, project_id: str):
    
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=int(project_id))

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseStatus.PROJECT_NOT_FOUND.value}
        )
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(
        project=project
    )

    if not collection_info:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseStatus.FETCHING_COLLECTION_INFO_FAILED.value}
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseStatus.FETCHING_COLLECTION_INFO_COMPLETED.value,
            "collection_info": collection_info
        })

@nlp_router.post("/search/{project_id}")
async def search_project(request: Request, project_id: str, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=int(project_id))

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseStatus.PROJECT_NOT_FOUND.value}
        )
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )


    results = await nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        top_k=search_request.top_k
    )

    

    if not results:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseStatus.SEARCH_FAILED.value}
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseStatus.SEARCH_COMPLETED.value,
            "results": [ result.dict()  for result in results ]
        })

@nlp_router.post("/answer/{project_id}")
async def answer_project(request: Request, project_id: str, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=int(project_id))

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseStatus.PROJECT_NOT_FOUND.value}
        )
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        top_k=search_request.top_k
    )

    if not answer:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseStatus.ANSWER_GENERATION_FAILED.value}
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseStatus.ANSWER_GENERATION_COMPLETED.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history 
        })









            