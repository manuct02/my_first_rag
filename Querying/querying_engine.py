import numpy as np
from sklearn.metrics.pairwise import cosine_similarity #es la función que establece relaciones de comparación entre matrices de vectores
from llama_index.embeddings.ollama import OllamaEmbedding #recuperamos el modelo de embeddings para vectorizar la query.


class QueryEngine:
    """
        Todo en esta clase viene de la carga en el disco del índice ya construido y cargado:
        - nodes: lista de nodos (Document) en el mismo orden que los embeddings
        - embeddings: matriz numpy de shape (num_nodos, dim)
        - embedding_model: nombre del modelo de Ollama que se usó al indexar
        """
    def __init__(self, nodes, embeddings, embedding_model: str = "nomic-embed-text"):
        self.nodes = nodes
        self.embeddings = embeddings
        self.embed_model = OllamaEmbedding(model_name=embedding_model)
    
    def embed_query(self, query_text: str):
        """convierte la pregunta ('str') en un vector 1D con float 32 para que ocupe menos memoria
        a la larga"""
        vec = self.embed_model.get_text_embedding(query_text)
        return np.array(vec, dtype=np.float32)
    
    def mejores_respuestas(self, query_text: str, k: int = 6):
        """
        Devuelve los k nodos más parecidos a la query.
        return: lista de tuplas (node, score)
        """ 
        
        """
        1)embedding de la pregunta, hay que convertir el vector de la prgunta en una matriz (aunque las
        demás dimensiones estén vacías) porque es lo que admite el cosine.similarity
        """
        q_vec = self.embed_query(query_text)   # shape (dim,)
        q_vec = q_vec.reshape(1, -1)            # shape (1, dim)

        """2) Similitud coseno contra todos los embeddings: esta función toma dos argumentos (dos matrices), y
        devuelve una lista de vectores. Cada vector tendrá tantos elementos como nodos haya, siendo estos la comparación 
        del embedding de la query con el embedding i-ésimo de los nodos. De esta forma, al tener sólo un vector para
        la query, sólo habrá un vector de salida, el que contenga las comparaciones de la query con todos y cada uno de
        los 300 y pico nodos"""
        
        similitudes = cosine_similarity(q_vec, self.embeddings)[0]  # shape (num_nodos,)

        """3) Índices de los k más grandes (de mayor a menor similitud): la gracia del RAG es obtener la información filtrada
         por coincidencias, es por ello que nos interesa encontrar dentro de ese vector de "similitudes" las posiciones
          de los números más altos, pues éstas reflejan las posiciones de los embeddings de información que más se 
           asemejan a la query """
        
        indices = np.argsort(similitudes)[::-1][:k] #argsort ordena los índices de la lista de menor a mayor, siendo así la posición del número más grande la última en aparecer, por eso le damos la vuelta 
        
        """Queremos de vuelta una lista con los fragmentos de texto de los embeddings recuperados por el cosine.similarity.
        Creamos una lista vacía"""
        resultados = []
        
        for idx in indices: #destacar que idx no está ordenado, si no que son los elementos de una lista donde el primer elemento es el número del nodo más parecido a la query en su lista de documents
            node = self.nodes[idx]
            score = float(similitudes[idx]) #recuperamos el valor de ese elemento para ver cómo de parecida es la query al embedding elegido para el contexto
            resultados.append((node.text, score))

        return resultados