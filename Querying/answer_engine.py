from typing import List, Tuple
import subprocess
from llama_index.llms.ollama import Ollama  


class AnswerEngine:

    def __init__(self, model_name: str = "qwen2.5:3b"):
        # Este nombre se usará para subprocess
        self.model_name = model_name

        # Creas tu objeto Ollama NORMAL (solo por compatibilidad)
        # No lo usaremos para la llamada final porque falla.
        self.llm = Ollama(model=model_name)


    def _build_prompt(self, question: str, context_chunks: List[str]) -> str:
        """
        Construye el prompt que verá el LLM.
        """

        context_chunks = [chunk if isinstance(chunk, str) else chunk[0] for chunk in context_chunks]
        contexto = "\n\n---\n\n".join(context_chunks)

        prompt = f"""Eres un asistente que responde en español usando el contexto dado.

Contexto:
{contexto}

Pregunta: {question}

Instrucciones:
- Si la respuesta está en el contexto, respóndela de forma clara y breve.
- Si no hay información suficiente, dilo explícitamente.
- No inventes datos que no aparezcan en el contexto.
- Estate muy atento a los nombres propios porque podrían ser importantes.

"""
        return prompt


    def answer(self, question: str, retrieved: List[Tuple[str, float]], max_chunks: int = 3) -> str:
        """
        Pregunta al LLM usando los mejores nodos recuperados.
        retrieved: lista de (texto_del_nodo, score)
        """

        # Nos quedamos con los k mejores textos
        context_chunks = retrieved[:max_chunks]

        # Construimos el prompt completo
        prompt = self._build_prompt(question, context_chunks)

        print("\n\n======= PROMPT QUE SE ENVIA A OLLAMA =======\n")
        print(prompt)
        print("\n============================================\n")

        # ==========================================
        # LLAMADA ROBUSTA A OLLAMA VIA SUBPROCESS
        # (igual que en terminal → siempre funciona)
        # ==========================================

        process = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        respuesta = process.stdout.decode("utf-8", errors="ignore")
        errores = process.stderr.decode("utf-8", errors="ignore")

        if errores.strip() != "":
            print("\n⚠️  [STDERR de Ollama]:\n", errores)

        return respuesta
