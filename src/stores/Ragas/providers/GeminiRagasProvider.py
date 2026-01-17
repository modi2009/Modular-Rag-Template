from ..RAGASLLMInterface import RAGASLLMInterface
import google.generativeai as genai
from ragas.llms import llm_factory
from ragas.embeddings.base  import embedding_factory
from ragas.metrics import faithfulness, context_precision
from ragas.metrics.collections import AnswerRelevancy
# providers/GeminiRagasProvider.py
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ResponseRelevancy  # Note: AnswerRelevancy is now ResponseRelevancy in latest versions
)

class GeminiRagasProvider(RAGASLLMInterface):
    def __init__(self, api_key: str):
        # Ragas 0.4+ uses a specific factory that handles Gemini natively
        self.generation_model_id = None
        self.embedding_model_id = None
        self.api_key = api_key
        self.llm = None
        self.gen_model = None
        genai.configure(api_key=api_key)

    def get_llm(self,model_id, system_instructions=""):
        self.gen_model = genai.GenerativeModel(model_id, system_instruction= system_instructions)
        self.llm =  llm_factory(model_id, provider = "google", client=self.gen_model)
        print("Initialized Gemini Ragas LLM with model:", model_id)
        return self.llm

    def get_embeddings(self,model_id):
        
        self.embeddings = embedding_factory("google", model = model_id, api_key = self.api_key)
        return self.embeddings

    def get_metrics(self):
        # Instantiate the objects and link your Gemini provider
        # This explicitly tells Ragas: "Use Gemini, NOT OpenAI defaults"
        
        m1 = Faithfulness(llm=self.llm)
        
        # Note: ResponseRelevancy requires both LLM and Embeddings
        m2 = ResponseRelevancy(llm=self.llm, embeddings=self.embeddings)
        
        m3 = ContextPrecision(llm=self.llm)
        
        # Return the list of INITIALIZED objects
        return [m1, m2, m3]