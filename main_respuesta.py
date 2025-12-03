from time import time
from Storer.loader_from_disk import LoaderFromDisk
from Querying.querying_engine import QueryEngine
from Querying.answer_engine import AnswerEngine
from time import time

start_time= time()

"""usamos la clase loader para no tener que llamar al ollama en cada query.
Asignamos nodes y embeddings a los archivos recuperados de los mismos."""

loader = LoaderFromDisk("/home/manuelmaturana/PRJ/RAG_Git/Storer/almacen")
nodes = loader.load_nodes()
embeddings = loader.load_embeddings()
config = loader.load_config()

"""clase query_engine para extraer las mejores respuestas que usará el LLM"""

query_engine= QueryEngine(nodes,embeddings,"nomic-embed-text")
#pregunta

question = "grandes genios del cine"

# Recuperar los 3 nodos más parecidos
retrieved = query_engine.mejores_respuestas(question, k=3)

#print("\n====== NODOS ELEGIDOS POR EL MOTOR DE QUERYING: ======\n")

#for i in range(len(retrieved)):
    #print("\n Nodo escogido {}: {}".format(i, retrieved[i]))

"""Answer_engine"""
answer_engine = AnswerEngine(model_name="qwen2.5:1.5b")

prompt= answer_engine._build_prompt(question, retrieved)

#print("\n====== PROMPT DEL LLM ======\n")
#print(prompt)

print("\nLa pregunta del usuario es:\n")
print(question)

final_answer = answer_engine.answer(question, retrieved, max_chunks=6)
print("\n====== RESPUESTA DEL LLM ======\n")
print(final_answer)

end_time= time()

total_time= end_time-start_time

print(f"Tiempo total transcurrido desde la pregunta hasta la respuesta: {total_time}")