# stores/ragas/providers/BaseRagasProvider.py
from abc import ABC, abstractmethod

class RAGASLLMInterface(ABC):
    @abstractmethod
    def get_llm(self):
        pass

    @abstractmethod
    def get_embeddings(self):
        pass

    @abstractmethod
    def get_metrics(self):
        pass