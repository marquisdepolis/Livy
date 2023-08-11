import concurrent.futures
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import List
import processing

MODEL = "gpt-3.5-turbo"
CHUNK_SIZE=2000

def base_gptcall(prompt):
    messages = [{"role": "system", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.1
    )
    return response.choices[0]['message']['content'].strip()

@retry(wait=wait_random_exponential(min=2, max=20), stop=stop_after_attempt(3), reraise=True)
def call_gpt(prompt):
    answers = []
    if len(prompt)>CHUNK_SIZE:
        textchunks = processing.split_text(prompt)
        for chunk in textchunks:
            answer = []
            # print(len(chunk))
            # print(chunk)
            answer = base_gptcall(chunk)
            answers.append(answer)
        return ' '.join(answers)
    else:
        return base_gptcall(prompt)

def recursive_analyze(text):
    text_chunks = processing.clean_text(text)
    text_chunks = processing.split_text(text_chunks)
    print("The total length of all text chunks is: ")
    print(len(text_chunks))
    # Use ThreadPoolExecutor to parallelize GPT calls
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for chunk in text_chunks:
            futures.append(executor.submit(call_gpt, f"Extract all insights, names and facts from the following text as would be useful for an investment memo:\n\n{chunk}"))
        insights_lists = [future.result() for future in futures]
    combined_insights = "\n".join(insights_lists)
    prompt = f"Please summarise. If no useful information is present, please reply with 'info not available':\n\n{combined_insights}"
    summary = call_gpt(prompt)
    return summary