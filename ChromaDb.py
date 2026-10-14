import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

class VectorMemoryManager:
    def __init__(self, persist_path: str, collection_name: str):
        # TODO: Inicializar el PersistentClient de ChromaDB con la ruta dada
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # TODO: Crear u obtener la colección usando self.client.get_or_create_collection
        # No olvides asignar una función de embedding por defecto
        self.collection = self.client.get_or_create_collection(collection_name)

    def upsert_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Implementa el método upsert para evitar errores por IDs duplicados.
        """
        # TODO: Usar self.collection.upsert
        self.collection.upsert(ids,documents,metadatas )

    def semantic_search(self, query_text: str, n_results: int = 3):
        """
        Realiza una búsqueda por similitud.
        Retorna los documentos más cercanos al query_text.
        """
        # TODO: Implementar query() y retornar los resultados
        self.collection.query(query_text, n_results)

# Ejemplo de prueba (puedes descomentar para probar localmente)
 manager = VectorMemoryManager("./my_local_db", "test_collection")
 manager.upsert_documents(["1"], ["Texto de prueba"], [{"cat": "general"}])
 print(manager.semantic_search("prueba"))
