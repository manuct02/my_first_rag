import os
import unicodedata
import fitz  # PyMuPDF
from llama_index.core import Document

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)

class Loader_robusto:
    def __init__(self, folder):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(base_dir, folder)

    def load(self):
        documents = []

        for filename in os.listdir(self.data_path):
            path = os.path.join(self.data_path, filename)

            # === TXT ===
            if filename.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                documents.append(
                    Document(text=normalize_text(text), metadata={"filename": filename})
                )

            # === PDF robusto con PyMuPDF ===
            elif filename.lower().endswith(".pdf"):
                try:
                    pdf = fitz.open(path)
                    text = ""
                    for page in pdf:
                        text += page.get_text("text") + "\n"

                    documents.append(
                        Document(text=normalize_text(text), metadata={"filename": filename})
                    )
                except Exception as e:
                    print(f"❌ Error leyendo {filename}: {e}")

        return documents