from .LLMEnums import LLMEnums

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):
        if provider == LLMEnums.GEMINI.value:
            from .providers.GEMINIProvider import GEMINIProvider
            return GEMINIProvider(
                api_key=self.config.GEMINI_API_KEY,
                default_input_max_tokens=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_output_max_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        return None