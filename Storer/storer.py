import os
import pickle #módulo estándar de Python que permite: guardar objetos python complejos en un archivo binario y recuperarlos después idénticos, con su estructura, tipos y contenido.
import numpy as np
import json #parecido a pickle pero para objetos sencillotes rollo 'str' o diccionarios.

class Storer:

    def __init__(self, nodes, embeddings, embeding_model= "nomic-embed-text"):

        self.nodes= nodes
        self.embeddings= embeddings
        self.embeding_model= embeding_model
    
    def save(self, almacen= "/home/manuelmaturana/PRJ/RAG/Storer/almacen"):

        """los nodos primero, usamos el pickle.dump que escribe el objeto (lista en este caso)
        en binario para posteriormente recuperarlo sin tener que cargarlo entero"""
        ruta_nodos= os.path.join(almacen, "nodes.pkl")
        with open(ruta_nodos, "wb") as f:
            pickle.dump(self.nodes, f)
        
        """los embeddings lo segundo"""
        ruta_embeddings= os.path.join(almacen, "embeddings.npz")
        np.savez(ruta_embeddings, embeddings= self.embeddings) #savez es como un zip, comprime varios arrays dentro codificados por nombres ("embeddings" representa lo que guardamos nosotros)

        """la configuración lo último"""
        ruta_config= os.path.join(almacen, "index_config.json" )
        config= {"embedding_model":self.embeding_model, "num_nodes": len(self.nodes), 
                 "embeding_dim": self.embeddings.shape}
        with open(ruta_config, "w", encoding="utf-8") as f:
            json.dump(config, f, indent= 4)
        
        print(f"[Storer] Archivos guardados en '{almacen}' correctamente.")