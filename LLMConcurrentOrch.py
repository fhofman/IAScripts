import asyncio
import logging
import os
import sys

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

assert sys.version_info >= (3, 12), "Requiere Python 3.12+"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TIMEOUT_S = 2          # timeout global de toda la ejecución
MAX_CONCURRENT = 2     # máximo de llamadas activas a la vez
TOTAL_CALLS = 10       # cantidad de llamadas a disparar

# Nombres de modelos configurables por variable de entorno
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "ai/llama3.2:1B-Q8_0")

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
llama_client = AsyncOpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",  # local, no se valida
)


# --- Llamadas a cada modelo (con latencia de red simulada) ---------------------

async def gpt_4_call(prompt: str) -> str:
    await asyncio.sleep(0.8)  # latencia simulada
    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout=TIMEOUT_S,
        )
    except openai.APITimeoutError as e:
        raise TimeoutError(f"OpenAI tardó más de {TIMEOUT_S}s") from e
    return response.choices[0].message.content or ""


async def claude_3_call(prompt: str) -> str:
    await asyncio.sleep(1.5)  # latencia simulada
    try:
        response = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            timeout=TIMEOUT_S,
        )
    except anthropic.APITimeoutError as e:
        raise TimeoutError(f"Claude tardó más de {TIMEOUT_S}s") from e
    return "".join(b.text for b in response.content if b.type == "text")


async def local_llama_call(prompt: str) -> str:
    await asyncio.sleep(2.5)  # latencia simulada: supera el timeout a propósito
    try:
        response = await llama_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout=TIMEOUT_S,
        )
    except openai.APITimeoutError as e:
        raise TimeoutError(f"Llama tardó más de {TIMEOUT_S}s") from e
    return response.choices[0].message.content or ""


# --- Corrutina envoltorio con semáforo -----------------------------------------

active = 0  # llamadas ejecutándose en este momento (para verificar el semáforo)


async def run_call(name: str, call, prompt: str, sem: asyncio.Semaphore) -> str:
    global active
    log.info("%-20s lanzada (esperando semáforo)", name)
    async with sem:
        active += 1
        log.info("%-20s INICIA   (activas: %d)", name, active)
        try:
            result = await call(prompt)
            log.info("%-20s termina -> %r", name, result)
            return result
        except asyncio.CancelledError:
            log.warning("%-20s cancelada por timeout global", name)
            raise
        finally:
            active -= 1


# --- Orquestación ---------------------------------------------------------------

async def main():
    prompt = "What is Kubernetes in 5 words?"
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    models = [gpt_4_call, claude_3_call, local_llama_call]

    tasks = {}
    for i in range(TOTAL_CALLS):
        model = models[i % len(models)]
        name = f"{model.__name__}#{i}"
        tasks[name] = asyncio.create_task(run_call(name, model, prompt, sem))

    try:
        async with asyncio.timeout(TIMEOUT_S):
            await asyncio.gather(*tasks.values(), return_exceptions=True)
    except TimeoutError:
        log.error("TIMEOUT GLOBAL: la ejecución superó %ss, se cancelan las pendientes", TIMEOUT_S)
        await asyncio.wait(tasks.values())  # esperar a que terminen de cancelarse

    # Distinguir respuestas exitosas de excepciones
    ok = errors = cancelled = 0
    for name, task in tasks.items():
        if task.cancelled():
            cancelled += 1
            log.warning("%-20s CANCELADA", name)
        elif (exc := task.exception()) is not None:
            errors += 1
            kind = "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
            log.error("%-20s %s: %s", name, kind, exc)
        else:
            ok += 1
            log.info("%-20s OK: %r", name, task.result())

    log.info("Resultado: %d OK, %d errores, %d canceladas", ok, errors, cancelled)


if __name__ == "__main__":
    asyncio.run(main())
