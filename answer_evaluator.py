import numpy as np
from llama_index.embeddings.ollama import OllamaEmbedding

class AnswerEvaluator:

    def __init__(self, embedding_model: str= "nomic-embed-text"):
        self.embed= OllamaEmbedding(model_name="nomic-embed-text")
    
    def _embed(self, text:str):
        return np.array(self.embed._get_text_embedding(text), dtype= np.float32)
    
    def exact_match(self, llm_answer: str, expected_answer: str):
        return(llm_answer.strip().lower()==expected_answer.strip().lower())
    
    def semantic_similarity(self, llm_answer: str, expected_answer: str):
        
        emb_llm= self._embed(llm_answer).reshape(1,-1)
        emb_answer= self._embed(expected_answer).reshape(-1,1)

        product= np.dot(emb_llm, emb_answer)
        denom= np.linalg.norm(emb_llm)*np.linalg.norm(emb_answer)
        sim= float(product/denom)
        
        return sim
    
    def faithfulness(self, llm_answer: str, context_nodes: str):
        emb_answer= self._embed(llm_answer).reshape(1,-1)
        sims=[]
        for ctx in context_nodes:
            emb_ctx= self._embed(ctx).reshape(1,-1)
            product= np.dot(emb_answer, emb_ctx.T)
            denom = np.linalg.norm(emb_answer) * np.linalg.norm(emb_ctx)
            sims.append(float(product / denom))
        return max(sims) if sims else 0
    
    def conciseness(self, llm_answer: str):
        length= len(llm_answer.split())

        if length<20:
            return 1
        if length>120:
            return 0
        else:
            return max(0.0, 1 - (length - 20) / 100)
    
    def evaluate(self, llm_answer: str, expected_answer: str, context_nodes: str):
        return {"EXACT MATCH": self.exact_match(llm_answer, expected_answer),
                "SEMANTIC SIMILARITY": self.semantic_similarity(llm_answer, expected_answer),
                "FAITHFULNESS": self.faithfulness(llm_answer, context_nodes),
                "CONCISENESS": self.conciseness(llm_answer)}


