import google.generativeai as genai
from ..LLMInterface import LLMInterface
from ..LLMEnums import GEMINIEnums
import logging
from typing import List, Union
import json

class GEMINIProvider(LLMInterface):

    def __init__(self, api_key: str, default_input_max_tokens: int = 2048, 
                 default_generation_output_max_tokens: int = 1024,
                 default_generation_temperature: float = 0.7):
        
        # Initialize the Google SDK with your API Key
        genai.configure(api_key=api_key)
        
        self.default_input_max_tokens = default_input_max_tokens
        self.default_output_max_tokens = default_generation_output_max_tokens
        self.default_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None
        
        # The 'Client' for Gemini is usually the model instance itself
        self.gen_model = None 

        self.enums = GEMINIEnums

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized GEMINIProvider and configured Google Generative AI.")

    def set_generation_model(self, model_id: str, system_instructions: str = ""):
        self.generation_model_id = model_id
        # We pre-initialize the GenerativeModel instance
        self.gen_model = genai.GenerativeModel(model_id, system_instruction= system_instructions)
        self.logger.info(f"Set generation model to {model_id}.")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Set embedding model to {model_id} with size {embedding_size}.")
    
    def process_text(self, text: str):
        """Simple truncation to avoid token limit errors."""
        return text[:self.default_input_max_tokens].strip()
    
    async def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                           temperature: float = None):
        """Generates text response using Gemini."""
        if self.gen_model is None:
            self.logger.error("Generation model is not set. Call set_generation_model first.")
            raise Exception("Generation model is not set.")

        self.logger.info(f"Generating text using {self.generation_model_id}.")

        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens or self.default_output_max_tokens,
            temperature=temperature or self.default_temperature
        )

        # Handle Chat vs Simple Completion
        # Gemini handles chat history as a list of {'role': 'user'|'model', 'parts': [str]}

        response = await self.gen_model.generate_content_async(prompt, generation_config=generation_config)
        
        if not response or not response.text:
            self.logger.error("No text returned from Gemini API.")
            return ""
            
        return response.text

    def embed_text(self, text: Union[str, List[str]], document_type: str = "retrieval_document"):
        """
        Generates embeddings. 
        Gemini 'text-embedding-004' supports tasks like 'retrieval_document' or 'retrieval_query'.
        """
        if self.embedding_model_id is None:
            self.logger.error("Embedding model is not set.")
            raise Exception("Embedding model is not set.")
        
        self.logger.info(f"Generating embeddings using {self.embedding_model_id}.")

        # If it's a single string, wrap it in a list
        input_text = [text] if isinstance(text, str) else text
        
        # Note: 'task_type' helps Gemini optimize the vector for RAG
        result = genai.embed_content(
            model=self.embedding_model_id,
            content=input_text,
            task_type=document_type
        )
        
        return result['embedding']

    async def rerank(self, query: str, documents: List, top_n: int = 10):
        if not documents:
            return []

        # Prepare the list of documents for the prompt
        # We use indices so Gemini can just return a list of numbers (efficient)
        doc_list_str = "\n".join([
            f"ID: {i} | Content: {doc.text[:500]}" # Limit text to save tokens
            for i, doc in enumerate(documents)
        ])

        prompt = f"""
        You are an expert search evaluator. Rank the following documents based on their relevance to the user query.
        
        Query: {query}
        
        Documents:
        {doc_list_str}
        
        Output only a JSON list of IDs in order of relevance, from most relevant to least. 
        Example: [3, 0, 2, 1]
        Return only the top {top_n} IDs.
        """

        try:
            # Call Gemini Flash (Fast & Free)
            response =  await self.gen_model.generate_content_async(prompt)
            # Clean response to ensure it's valid JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            relevant_indices = json.loads(cleaned_response)
            print(f"Reranking result indices: {relevant_indices}")

            # Map indices back to objects
            return [documents[i] for i in relevant_indices if i < len(documents)]
        except Exception as e:
            print(f"Reranking failed: {e}")
            return documents[:top_n] # Fallback to original order
    

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "parts": [{"text": prompt}],
        }