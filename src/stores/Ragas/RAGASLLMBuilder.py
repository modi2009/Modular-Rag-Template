from .providers.GeminiRagasProvider import GeminiRagasProvider

class RagasFactory:
    def __init__(self, config: dict):
        self.config = config

    def get_provider(self,provider_type: str):
        if provider_type.lower() == "google":
            return GeminiRagasProvider(api_key= self.config.GEMINI_API_KEY)
        # Add other providers here (openai, anthropic, etc.)
        raise ValueError(f"Provider {provider_type} not supported for Ragas.")