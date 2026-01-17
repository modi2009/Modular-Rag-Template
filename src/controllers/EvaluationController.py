import pandas as pd
from ragas import aevaluate
from datasets import Dataset
from .BaseController import BaseController
from .NLPController import NLPController

class EvaluationController(BaseController):
    def __init__(self, nlp_controller: NLPController, ragas_provider):
        super().__init__()
        self.nlp_controller = nlp_controller
        
        # These come from your GeminiRagasProvider (RAGASLLMInterface)
        self.ragas_provider = ragas_provider
        self.eval_llm = ragas_provider.llm
        self.eval_embeddings = ragas_provider.embeddings
        self.metrics = ragas_provider.get_metrics()

    async def run_evaluation_batch(self, project, test_queries: list):
        """
        Runs a full evaluation on a list of test queries.
        """
        results_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [] 
        }

        for query in test_queries:
            # 1. Run Search
            retrieved_docs = await self.nlp_controller.search_vector_db_collection(
                project=project, text=query
            )
            
            # 2. Generate Answer
            answer, _, _ = await self.nlp_controller.answer_rag_question(
                project=project, query=query
            )

            # 3. Append data (Note: Ragas v0.4+ uses 'question', 'answer', 'contexts')
            results_data["question"].append(query)
            results_data["answer"].append(answer)
            results_data["contexts"].append([doc.text for doc in retrieved_docs])
            results_data["ground_truth"].append("Reference answer if available") 

        # 4. Convert to Dataset
        dataset = Dataset.from_dict(results_data)
        print("Evaluation dataset prepared with", len(dataset), "entries.")

        # 5. Perform Evaluation
        # We pass the factory-initialized LLM and Embeddings directly
        result = await aevaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings
        )
        print("Evaluation completed.")

        return result.to_pandas()