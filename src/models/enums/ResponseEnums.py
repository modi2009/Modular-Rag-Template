from enum import Enum

class ResponseStatus(Enum):
    FILE_VALIDATED_SUCCESS = "file_validate_successfully"
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    FILE_PROCESSING_STARTED = "file_processing_started"
    FILE_PROCESSING_COMPLETED = "file_processing_completed"
    PROJECT_NOT_FOUND = "project_not_found"
    INDEXING_FAILED = "indexing_failed"
    INDEXING_COMPLETED = "indexing_completed"
    FETCHING_COLLECTION_INFO_FAILED = "fetching_collection_info_failed"
    FETCHING_COLLECTION_INFO_COMPLETED = "fetching_collection_info_completed"
    SEARCH_FAILED = "search_failed"
    SEARCH_COMPLETED = "search_completed"
    ANSWER_GENERATION_FAILED = "answer_generation_failed"
    ANSWER_GENERATION_COMPLETED = "answer_generation_completed"