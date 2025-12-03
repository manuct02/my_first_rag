from llama_index.core.ingestion import IngestionPipeline #aplica las transformaciones del argumento al objeto en cuestión en el orden en el que aparecen
from llama_index.core.node_parser import TokenTextSplitter #separa los elementos de un "Document.text" en la cantidad de nodos especificada
from llama_index.core import Document
import unicodedata
from Loading.loader import normalize_text
class Transformer:
    def __init__(self, documents_raw):
        self.documents_raw = documents_raw

        self.pipeline = IngestionPipeline(
            transformations=[TokenTextSplitter(chunk_size=600, chunk_overlap=100)])

    def nodes(self):
        clean_docs = []

        for doc in self.documents_raw:
            if not doc.text or doc.text.strip() == "":
                continue  # 🔥 documento vacío -> fuera

            normalized = normalize_text(doc.text)
            clean_docs.append(Document(text=normalized, metadata=doc.metadata))

        return self.pipeline.run(documents=clean_docs)