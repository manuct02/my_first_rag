from Storer.loader_from_disk import LoaderFromDisk
from Querying.querying_engine import QueryEngine
from retrieval_evaluator import RetrievalEvaluator
from loader_data import DataLoader

loader= LoaderFromDisk("/home/manuelmaturana/PRJ/RAG_Git/Storer/almacen")

nodes= loader.load_nodes()
embeddings= loader.load_embeddings()
config= loader.load_embeddings()

query_engine= QueryEngine(nodes, embeddings, "nomic-embed-text")
eval_dataset= DataLoader('/home/manuelmaturana/PRJ/RAG_Git/eval_dataset.json')
evaluator= RetrievalEvaluator(query_engine, 5)

print('====== RESULTADOS DEL EVALUADOR DEL RETRIEVAL ======')

for question, expected in eval_dataset.itir():
    print("\n Pregunta: '{}' ".format(question))
    print("\n Resultado esperado de algún nodo: '{}'\n".format(expected))
    result= evaluator.evaluate_question(question,expected)
    print(result)
