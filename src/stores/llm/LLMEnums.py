from enum import Enum

class LLMEnums(Enum):
    GEMINI = "GEMINI"

class GEMINIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "model"


class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"