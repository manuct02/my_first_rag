import numpy as np
import numpy as np
from llama_index.embeddings.ollama import OllamaEmbedding #contiene lo referente a modelos de Ollama, concretamente el de embeddings
from tqdm import tqdm
embed_model = OllamaEmbedding(model_name="nomic-embed-text") #especificamos el modelo a usar, en este caso el "nomic-embed-text"


class Indexer_robusto:
    """
    Recibe nodos, genera sus embeddings (vectores), monta una matriz
    y permite hacer búsqueda semántica top-k.
    """

    def __init__(self, nodos):
        self.nodos = nodos

        # Paso 1: generar embeddings
        self._ensure_embeddings()

        # Paso 2: matriz numpy
        self.embeddings = self._build_matrix()


    def _ensure_embeddings(self):
        for node in tqdm(self.nodos, desc="🟦 Generando embeddings"):
            if node.embedding is None:
                node.embedding = embed_model.get_text_embedding(node.text)


    def _build_matrix(self):
        """Crea la matriz numpy de embeddings."""
        matrix = []
        for node in self.nodos:
            matrix.append(node.embedding)
        return np.array(matrix)
    
    def get_matrix(self):
        """Devuelve la matriz numpy de embeddings."""
        return self.embeddings

    def get_embeddings_list(self):
        """Devuelve la lista de embeddings tal cual (no numpy)."""
        return [node.embedding for node in self.nodos]
