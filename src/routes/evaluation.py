# routes/evaluation.py
from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from models.ProjectModel import ProjectModel
from controllers import NLPController, EvaluationController
from models import ResponseStatus

import logging

logger = logging.getLogger('uvicorn.error')

eval_router = APIRouter(
    prefix="/evaluation",
    tags=["evaluation"],
)

class EvaluationRequest(BaseModel):
    test_queries: List[str]

@eval_router.post("/{project_id}")
async def run_project_evaluation(request: Request, project_id: str, eval_request: EvaluationRequest):
    """
    Runs a Ragas evaluation batch for a specific project.
    """
    
    # 1. Validate Project
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_by_id(project_id=int(project_id))
    
    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseStatus.PROJECT_NOT_FOUND.value}
        )

    # 2. Initialize Controllers
    # We need NLPController because EvaluationController depends on it
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser
    )

    # Note: request.app.ragas_provider was initialized in main.py startup
    evaluation_controller = EvaluationController(
        nlp_controller=nlp_controller,
        ragas_provider=request.app.ragas_provider
    )

    try:
        logger.info(f"Starting evaluation for project {project_id} with {len(eval_request.test_queries)} queries.")
        
        # 3. Run Evaluation
        report_df = await evaluation_controller.run_evaluation_batch(
            project=project,
            test_queries=eval_request.test_queries
        )

        # 4. Convert Pandas DataFrame result to list of dicts for JSON response
        report_data = report_df.to_dict(orient="records")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "signal": "EVALUATION_COMPLETED",
                "project_id": project_id,
                "metrics": report_data
            }
        )

    except Exception as e:
        logger.error(f"Evaluation failed for project {project_id}: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "EVALUATION_FAILED",
                "error": str(e)
            }
        )