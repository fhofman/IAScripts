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
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError(
                f"Largo distinto: ids={len(ids)}, documents={len(documents)}, metadatas={len(metadatas)}"
            )
        # TODO: Usar self.collection.upsert
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def semantic_search(self, query_text: str, n_results: int = 3):
        """
        Realiza una búsqueda por similitud.
        Retorna los documentos más cercanos al query_text.
        """
        total = self.collection.count()
        if total == 0:
            return []
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, total),
        )
        return results["documents"][0]

# Ejemplo de prueba (puedes descomentar para probar localmente)
manager = VectorMemoryManager("./my_local_db", "test_collection")


    # 0. Arrancar con la colección vacía (por si quedaron datos de corridas anteriores)
existentes = manager.collection.get()["ids"]
if existentes:
    manager.collection.delete(ids=existentes)
assert manager.collection.count() == 0
assert manager.semantic_search("algo") == [], "Con la colección vacía debería devolver []"
print("✅ Colección vacía")

# 1. Inserción
datos = [
    ("1", "El gato duerme mucho y caza ratones de noche.", "animales"),
    ("2", "Los perros necesitan pasear dos veces al día.", "animales"),
    ("3", "Python es un lenguaje de programación para ciencia de datos.", "tecnologia"),
    ("4", "Docker empaqueta aplicaciones en contenedores.", "tecnologia"),
    ("5", "El asado se cocina lento a la parrilla con leña.", "cocina"),
    ("6", "Messi ganó el Mundial de fútbol en 2022.", "deportes"),
]
manager.upsert_documents(
    [d[0] for d in datos], [d[1] for d in datos], [{"cat": d[2]} for d in datos]
)
assert manager.collection.count() == len(datos), "No se insertaron todos los documentos"
print(f"✅ Inserción: {manager.collection.count()} documentos")

# 2. Upsert de un ID existente: actualiza sin duplicar
nuevo = "El gato es un felino independiente que duerme al sol."
manager.upsert_documents(["1"], [nuevo], [{"cat": "animales"}])
assert manager.collection.count() == len(datos), "El upsert duplicó el documento"
assert manager.collection.get(ids=["1"])["documents"][0] == nuevo, "El upsert no actualizó el texto"
print("✅ Upsert actualiza sin duplicar")

# 3. n_results se respeta
assert len(manager.semantic_search("cualquier cosa", n_results=2)) == 2
print("✅ n_results respetado")

# 4. Búsqueda semántica (chequeo blando: depende del modelo de embeddings)
categoria = {d[1]: d[2] for d in datos}
categoria[nuevo] = "animales"
consultas = [
    ("mascotas en casa", "animales"),
    ("lenguaje para programar", "tecnologia"),
    ("receta de carne", "cocina"),
    ("campeonato de fútbol", "deportes"),
]
for texto, esperada in consultas:
    top = manager.semantic_search(texto, n_results=1)[0]
    obtenida = categoria[top]
    print(f"{'✅' if obtenida == esperada else '⚠️'} '{texto}' -> {obtenida} | {top}")
