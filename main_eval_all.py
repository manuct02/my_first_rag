from evaluation_loop import run_full_evaluation



run_full_evaluation(
    index_path="/home/manuelmaturana/PRJ/RAG_Git/Storer/almacen",
    dataset_path="/home/manuelmaturana/PRJ/RAG_Git/eval_dataset.json",
    k=5,
    out_file="resultados_finales.jsonl"
)
