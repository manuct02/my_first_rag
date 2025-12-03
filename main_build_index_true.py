from Loading.loader import Loader_robusto
from Indexing.transformer import Transformer
from Indexing.indexer import Indexer_robusto
from Storer.storer import Storer
from time import time
from Storer.loader_from_disk import LoaderFromDisk
from llama_index.llms.ollama import Ollama

start_time= time()

from llama_index.embeddings.ollama import OllamaEmbedding

print("\n⏳ Cargando modelo de embeddings SOLO UNA VEZ…")
try:
    emb = OllamaEmbedding(model_name="nomic-embed-text")
    emb.get_text_embedding("hola")
    print("✔️ Modelo de embeddings precalentado\n")
except Exception as e:
    print("❌ FALLO precalentando embeddings:", e)
    exit()

"""
cargamos los documentos de la carpeta deseada con la clase Loader en un objeto document:
- input: PDF's, SQL, word, .exl,...
- output: lista de Document "documents"
"""

document = Loader_robusto("data_quantum")
documents_list= document.load()
print("Documentos cargados:", len(documents_list))
"""
la clase transformafdor convierte los "Documents" en bruto en nodos más pequeños que sean vectorizables:
- input: lista de Document
- output: lista de nodos "nodos"
"""

transformador = Transformer(documents_list)
nodos = transformador.nodes() #lista de "Documents" más manejables
print("Nodos generados:", len(nodos))

"""
la clase indexer vectoriza los nodos a partir de un modelo elegido (ollama en este caso: 'nomic-embed-text'):
- input: lista de nodos
- output: lista de embeddings
"""
index= Indexer_robusto(nodos) #creamos un objeto Indexer al que le atribuimos nuestra lista de nodos y lo convierte
lista_de_embeddings = index.get_embeddings_list() #obtenemos la lista con todos los embeddings vecorizados
embeddings_matrix = index.get_matrix() #la matriz de: (número de embeddings x longitud de los vectores)-dimensiones
numero_de_embeddings = len(lista_de_embeddings) #cantidad de embeddings(==nodos)

"""
Como runear todo el modelo para los embeddings cada vez que necesitemos los vectores de los 
300 y pico nodos es un coñazo...usamos la clase Storer para cargarlos en el disco y guardar
el modelo de embeddings, los nodos en bruto, y los nodos vectorizados en la matriz (que usaremos
para la query)
"""
store= Storer(nodos, embeddings_matrix, embeding_model= "nomic-embed-text")
mi_creacion= store.save("/home/manuelmaturana/PRJ/RAG/Storer/almacen_quantum") #el .save es una función de la clase Storer que guarda los argumentos del objeto storer en el directorio del argumento

"""
una vez almacenados en una carpeta nos olvidamos de storer y cargamos desde el disco usando
loader_from_disk
"""

end_time= time()
total_time= end_time-start_time
print(total_time, numero_de_embeddings)
