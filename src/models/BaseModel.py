from helpers import Settings, get_settings

class BaseModel:
    def __init__(self, db_client: object):
        self.db_client = db_client
        self.settings: Settings = get_settings()