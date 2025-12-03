import os
import pickle
import json
import numpy as np

class LoaderFromDisk:

    def __init__(self, almacen: str):

        self.almacen= almacen

    def load_nodes(self):

        """cargamos con esta función todo"""

        """asignamos a los nodos el archivo .pkl"""
        ruta_nodes= os.path.join(self.almacen, "nodes.pkl")
        with open(ruta_nodes, "rb") as f:
            nodes = pickle.load(f)
        return nodes
    
    def load_embeddings(self):
        ruta_embeddings= os.path.join(self.almacen, "embeddings.npz")
        data= np.load(ruta_embeddings)
        embeddings = data["embeddings"]
        return embeddings
    
    def load_config(self):
        ruta_config= os.path.join(self.almacen, "index_config.json")
        with open(ruta_config, "r", encoding="utf-8") as f:
            config= json.load(f)
        return config
    