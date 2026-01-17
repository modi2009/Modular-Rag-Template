from .BaseController import BaseController
from .ProjectController import ProjectController
from models import ResponseStatus
from fastapi import UploadFile
import re
import os


class DataController(BaseController):
    def __init__(self):
        
        super().__init__()
        self.scale_mb = 1024*1024
        self.scale_kb = 1024
    

    def validate_file(self, file: UploadFile) -> bool:
        # Implement file validation logic here
        print(file.content_type)
        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseStatus.FILE_TYPE_NOT_SUPPORTED.value
        if file.size > self.app_settings.FILE_MAX_SIZE * self.scale_mb:
            return False, ResponseStatus.FILE_SIZE_EXCEEDED.value
        
        return True, ResponseStatus.FILE_VALIDATED_SUCCESS.value
    
    def get_clean_file_name(self, orig_file_name: str):

        # remove any special characters, except underscore and .
        cleaned_file_name = re.sub(r'[^\w.]', '', orig_file_name.strip())

        # replace spaces with underscore
        cleaned_file_name = cleaned_file_name.replace(" ", "_")

        return cleaned_file_name
    
    def generate_unique_file_path(self, project_id: str, file_name: str) -> str:
        
        clean_file_name = self.get_clean_file_name(file_name)
        random_suffix = self.generate_random_string(12)
        project_path = ProjectController().get_project_files_dir(project_id)

        unique_file_path = os.path.join(
            project_path,
            f"{random_suffix}_{clean_file_name}"
        )

        while os.path.exists(unique_file_path):
            random_suffix = self.generate_random_string(12)
            unique_file_path = os.path.join(
                project_path,
                f"{random_suffix}_{clean_file_name}"
            )
        return unique_file_path, random_suffix + "_" + clean_file_name

