import numpy as np
from Querying.querying_engine import QueryEngine

class RetrievalEvaluator:
    def __init__(self, query_engine: QueryEngine, k=5):
        self.query_engine = query_engine
        self.k = k

    def evaluate_question(self, question, expected_substring):
        retrieved= self.query_engine.mejores_respuestas(question)
        retrieved_texts= [text for text, score in retrieved]
        #------------------MÉTRICAS------------------------------
        """Recal@k---> algún nodo contiene la cadena esperada??"""
        recall= any(expected_substring.lower() in txt.lower() for txt in retrieved_texts)

        counter= 0

        for i in range(len(retrieved_texts)):
            if (expected_substring.lower() in retrieved_texts[i].lower())==True:
                counter+=1
        """Precision@k---> proporción de nodos relevantes encontrados"""
        relevant_count= sum(expected_substring.lower() in txt.lower() for txt in retrieved_texts)
        precision= counter/self.k
        """MMR---> reciprocal rank de la primera aparición"""
        rr= 0
        for rank, txt in enumerate(retrieved_texts, start=1):
            if expected_substring.lower() in txt.lower():
                rr=1/rank
                break
        
        return {"recall@k": recall,
            "precision@k": precision,
            "mrr": rr}

    
