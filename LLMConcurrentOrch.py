import asyncio
import os
from openai import AsyncOpenAI, APITimeoutError
from anthropic import AsyncAnthropic


openai =  AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic =  AsyncAnthropic(api_key=os.getenv("ANTRHROPIC_API_KEY"))
llama = AsyncOpenAI(
    base_url="http://localhost:12434/engines/v1",
    api_key="not-needed",  # local, no se valida
)

async def concurrent_models(prompt):
    # Programamos las tres llamadas para que ocurran al mismo tiempo
    results = await asyncio.gather(
        gpt_4_call(prompt),
        claude_3_call(prompt),
        local_llama_call(prompt),
        return_exceptions=True # Crucial para que un fallo no tumbe todo
    )
    return results

async def gpt_4_call(prompt):
    try:
        response = await openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            timeout=3
        )
        await asyncio.sleep(3)
        print('openai')
        print(response.choices[0].message.content or "")
        return response.choices[0].message.content or ""
    except TimeoutError:
        raise TimeoutError("La llamada a OpenAI tardó más de 2 segundos")
    return response.content[0].text

async def claude_3_call(prompt):
    try:
        response = await anthropic.messages.create(
            model="claude-sonnet-5-5",
            messages=[{"role": "user", "content": prompt}],
            timeout=3,
            max_tokens=100,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        print("Anthropic")
        print(text)
        return text
    except APITimeoutError:
        raise TimeoutError("La llamada a Claude tardó más de 3 segundos")   
    

async def local_llama_call(prompt):
    try:
        response = await llama.chat.completions.create(
            model="ai/llama3.2:1B-Q8_0",
            messages=[{"role": "user", "content": prompt}],
            timeout=3,
        )
        await asyncio.sleep(1)
        content = response.choices[0].message.content
        print('llama : ')
        print(content)
        return content
    except APITimeoutError:
        raise TimeoutError("La llamada a llama tardó más de 2 segundos")   



async def main():
    prompt = "What is Kubernetes in 5 words?"
    results = await concurrent_models(prompt)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())