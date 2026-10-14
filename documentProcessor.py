import re
import tiktoken
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, model_encoding: str = "cl100k_base"):
        # TODO: Inicializar el encoding de tiktoken
        self.tokenizer = tiktoken.get_encoding(model_encoding)
        
        # TODO: Configurar RecursiveCharacterTextSplitter
        # Debe usar una función personalizada para medir longitud por TOKENS, no caracteres.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,          # máximo de TOKENS por chunk
            chunk_overlap=50,        # solapamiento en TOKENS
            length_function=self.calculate_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def clean_text(self, text: str) -> str:
        """Limpia el texto eliminando espacios duplicados y saltos de línea innecesarios."""
        # IMPLEMENTA AQUÍ: Usa regex para limpiar el texto
        text = text.replace("\r\n", "\n").replace("\r", "\n")   # normaliza saltos de línea
        text = re.sub(r"[ \t\f\v]+", " ", text)                  # espacios/tabs múltiples -> uno
        text = re.sub(r" *\n *", "\n", text)                     # quita espacios alrededor de \n
        text = re.sub(r"\n{3,}", "\n\n", text)                   # máx. una línea en blanco seguida
        return text.strip()

    def calculate_tokens(self, text: str) -> int:
        """Calcula la cantidad de tokens usando el tokenizer de tiktoken."""
        # IMPLEMENTA AQUÍ
        if not text:
            return 0
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def process_document(self, raw_text: str) -> List[str]:
        """Pipeline: Limpieza -> Fragmentación."""
        # 1. Limpiar el texto
        cleaned_text = self.clean_text(raw_text)
        if not cleaned_text:
            return []
        # TODO: 2. Aplicar el splitter
        chunks = self.splitter.split_text(cleaned_text)
        
        # TODO: 3. Retornar los chunks procesados

        return [chunk.strip() for chunk in chunks if chunk.strip()]

if __name__ == "__main__":
    # Texto de prueba
    text = "Plic era una pequeña gota de lluvia que vivía en una nube blanca como el algodón. Le encantaba vivir allí mientras miraba el vasto mundo multicolor que se extendía a sus pies. Sentía una extraña atracción por ese suelo lejano y, muy en secreto, anhelaba caer."
    processor = DocumentProcessor()
    chunks = processor.process_document(text)
    print(f"Chunks generados: {len(chunks)}")

