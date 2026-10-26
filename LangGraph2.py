import asyncio
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 1. Define el Estado (State) del agente
class AgentState(TypedDict):
    # TODO: Usa add_messages para que el historial sea acumulativo
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Herramientas
def get_weather(location: str):
    """Consulta el clima actual."""
    return "Soleado, 22°C"

tools = [get_weather]
# TODO: Crea el ToolNode con la lista de herramientas
tool_node = ToolNode(tools)

# 3. LLM con Herramientas vinculadas
# TODO: Instancia ChatOpenAI y usa .bind_tools(tools)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# 4. Nodos
async def call_model(state: AgentState):
    # TODO: Invoca al modelo con los mensajes actuales
    # Retorna un diccionario con la nueva clave "messages"
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}

# 5. Orquestación del Grafo
workflow = StateGraph(AgentState)

# TODO: Añade el nodo 'agent' y el nodo 'tools'
# TODO: Define la lógica de aristas:
# - START -> agent
# - agent -> conditional_edges (usa tools_condition)
# - tools -> agent
workflow.add_node("agent", call_model) 
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

# 6. Persistencia
# TODO: Instancia MemorySaver()
memory = MemorySaver()

# TODO: Compila el grafo pasando el checkpointer
app = workflow.compile(checkpointer=memory)

async def main():
    # Identificador de sesión para persistencia
    thread_config = {"configurable": {"thread_id": "dev_test_01"}}
    
    # Primera interacción
    print("--- Paso 1 ---")
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="Que clima hace en Madrid?")]},
        config=thread_config,
    )
    print(result["messages"][-1].content)
    
    # Segunda interacción (reanudando desde checkpoint)
    print("--- Paso 2 ---")
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="Que clima hace en Londres?")]},
        config=thread_config,
    )
    print(result["messages"][-1].content)
    
if __name__ == "__main__":
    asyncio.run(main())
