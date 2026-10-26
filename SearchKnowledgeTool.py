from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Esquema de entrada de la herramienta: el LLM lee estas descripciones para completar los argumentos
class SearchInput(BaseModel):
    query: str = Field(description="La consulta de búsqueda del usuario, en lenguaje natural")
    limit: int = Field(default=3, description="Cantidad máxima de resultados a devolver (entre 1 y 10)", ge=1, le=10)


# Simulación de base de datos vectorial
class MockVectorDB:
    async def similarity_search(self, query: str, k: int):
        return [f"Resultado para {query}"]

db = MockVectorDB()


@tool(args_schema=SearchInput)
async def search_knowledge_base(query: str, limit: int = 3) -> Union[List[str], str]:
    """
    Busca en la base de conocimiento de soporte los artículos más relevantes para la consulta del usuario.
    Usala cuando necesites información para responder una pregunta de soporte
    (por ejemplo: recuperar la contraseña, facturación o configuración de la cuenta).
    Devuelve una lista con los textos de los artículos encontrados, ordenados por relevancia,
    o un mensaje que empieza con "ERROR:" si la búsqueda falla.
    """
    try:
        resultados = await db.similarity_search(query, k=limit)
        return resultados
    except Exception as e:
        return f"ERROR: no se pudo buscar en la base de conocimiento ({type(e).__name__}: {e}). Intentá reformular la consulta o derivar a soporte humano."


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(search_knowledge_base.ainvoke({"query": "Perdí mi contraseña", "limit": 2})))
