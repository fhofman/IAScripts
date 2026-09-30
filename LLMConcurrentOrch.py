import asyncio
import os
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


openai =  AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic =  AsyncAnthropic(api_key=os.getenv("ANTRHROPIC_API_KEY"))

async def concurrent_models(prompt):
    # Programamos las tres llamadas para que ocurran al mismo tiempo
    results = await asyncio.gather(
        gpt_4_call(prompt),
        claude_3_call(prompt),
        #local_llama_call(prompt),
        return_exceptions=True # Crucial para que un fallo no tumbe todo
    )
    return results

async def gpt_4_call(prompt):
    try:
        response = await openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            timeout=2
        )
        #await asyncio.sleep(3)
        print('openai')
        print(response.choices[0].message.content or "")
        return response.choices[0].message.content or ""
    except TimeoutError:
        raise TimeoutError("La llamada a OpenAI tardó más de 2 segundos")
    return response.content[0].text

async def claude_3_call(prompt):
    try:
        response = await anthropic.messages.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": prompt}],
            timeout=3,
            max_tokens=100,
        )
        #await asyncio.sleep(2)
        print('Anthropic')
        print(response.content.text)
        return response.content.text
    except TimeoutError:
        raise TimeoutError("La llamada a Claude tardó más de 2 segundos")
    return response.content[0].text   
    

# async def local_llama_call(prompt):
#     try:
#         response = llama.ChatCompletion.create(
#             model="llama-3.1-8b-instant",
#             messages=[{"role": "user", "content": prompt}],
#             timeout=2
#         )
#         await asyncio.sleep(1)
#         print(response.choices[0].message.content)
#         return response.choices[0].message.content
#     except TimeoutError:
#         raise TimeoutError("La llamada a llama tardó más de 2 segundos")
#     return response.content[0].text    



async def main():
    prompt = "What is the capital of France?"
    results = await concurrent_models(prompt)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())