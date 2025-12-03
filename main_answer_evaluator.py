from Storer.loader_from_disk import LoaderFromDisk
from Querying.querying_engine import QueryEngine
from Querying.answer_engine import AnswerEngine
from answer_evaluator import AnswerEvaluator
from loader_data import DataLoader

load= LoaderFromDisk("/home/manuelmaturana/PRJ/RAG_Git/Storer/almacen")
nodes= load.load_nodes()
embeddings= load.load_embeddings()
config= load.load_config()

query_engine= QueryEngine(nodes, embeddings, "nomic-embed-text")
answer_engine= AnswerEngine("qwen2.5:1.5b")

dataset= DataLoader('/home/manuelmaturana/PRJ/RAG_Git/eval_dataset.json')

evaluator= AnswerEvaluator("nomic-embed-text")

print("⏳ Cargando Qwen en Ollama una vez...")
answer_engine.llm.complete("Hola", timeout=200)
print("✔️ Modelo cargado.")


for question, expected in dataset.itir():
    retrieved= query_engine.mejores_respuestas(question, 5)
    retrieved_texts= [txt for txt, score in retrieved]

    llm_answer= answer_engine.answer(question, retrieved_texts, 5)

    metrics= evaluator.evaluate(llm_answer, expected, retrieved_texts)

    print("\n-PREGUNTA:", question)
    print("\n 'ESPERADO':", expected)
    print("\n MÉTRICAS:", metrics)
