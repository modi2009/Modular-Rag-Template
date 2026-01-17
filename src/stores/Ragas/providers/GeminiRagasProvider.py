from ..RAGASLLMInterface import RAGASLLMInterface
import google.generativeai as genai
from ragas.llms import llm_factory
from ragas.embeddings.base  import embedding_factory
from ragas.metrics import faithfulness, context_precision
from ragas.metrics.collections import AnswerRelevancy

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
        # Attach the LLM/Embeddings to the metrics directly
        # This prevents Ragas from trying to use OpenAI defaults
        m1 = faithfulness
        m1.llm = self.llm
        
        m2 = AnswerRelevancy
        m2.llm = self.llm
        m2.embeddings = self.embeddings
        
        m3 = context_precision
        m3.llm = self.llm
        
        return [m1, m2, m3]


