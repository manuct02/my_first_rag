import json
from loader_data import DataLoader
from Storer.loader_from_disk import LoaderFromDisk
from Querying.querying_engine import QueryEngine
from Querying.answer_engine import AnswerEngine
from retrieval_evaluator import RetrievalEvaluator
from answer_evaluator import AnswerEvaluator



def run_full_evaluation(index_path, dataset_path, k=5, out_file="evaluation_results.jsonl"):

    # ==== Cargar índice desde disco ====
    loader = LoaderFromDisk(index_path)
    nodes = loader.load_nodes()
    embeddings = loader.load_embeddings()
    config = loader.load_config()

    # ==== Motores RAG ====
    query_engine = QueryEngine(nodes, embeddings, "nomic-embed-text")
    answer_engine = AnswerEngine("qwen2.5:1.5b")

    # ==== PRECALENTAR EL MODELO ====
    print("\n⏳ Cargando el modelo Qwen en Ollama SOLO UNA VEZ...")
    try:
        answer_engine.llm.complete("hola", timeout=200)
        print("✔️ Modelo cargado correctamente.\n")
    except Exception as e:
        print("❌ Error al precalentar el modelo:", e)
        return
    
    retrieval_eval= RetrievalEvaluator(query_engine=query_engine, k=5)
    answer_eval= AnswerEvaluator(embedding_model="nomic-embed-text")

    dataset= DataLoader(dataset_path)

    file_out= open(out_file, "w", encoding="utf-8")

    for question, expected in dataset.itir():
        """Evaluación del retrieval"""
        retrieved= query_engine.mejores_respuestas(query_text=question, k=5)
        retrieved_text= [txt for txt, score in retrieved]

        r_metrics= retrieval_eval.evaluate_question(question=question, expected_substring=expected)

        """Evaluación de las respuestas del LLM"""
        llm_answer= answer_engine.answer(question=question, retrieved=retrieved_text, max_chunks=5)
        
        a_metrics= answer_eval.evaluate(llm_answer=llm_answer, expected_answer=expected, context_nodes=retrieved_text)

        """Guardamos los registros en el formaton JSONL"""
        record= {"PREGUNTA": question,
                 "EXPECTED": expected,
                 "MÉTRICAS DEL RETRIEVAL": r_metrics,
                 "RESPUESTA DEL LLM": llm_answer,
                 "MÉTRICAS DEL LA RESPUESTA": a_metrics,
                 "FRAGMENTOS USADOS": retrieved_text}
        
        file_out.write(json.dumps(record, ensure_ascii=False)+ "\n")
    
    file_out.close()
    print("\n🎯 Evaluación completa. Resultados guardados en:", out_file)

