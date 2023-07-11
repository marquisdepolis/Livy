import pickle
import PIL
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import os
import csv
import warnings
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple, Dict
from urllib.parse import urlparse, urljoin

import faiss
import numpy as np
import openai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tkinter import Tk, filedialog
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from readppt import read_ppt
import readpdf
import readdoc
import processing
import embeddings
import gpt_calls
import urlscrape

load_dotenv()
warnings.filterwarnings("ignore")
openai.api_key = os.getenv('OPENAI_API_KEY')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
MODEL = "gpt-3.5-turbo"
CHUNK_SIZE=2000
CHUNK_INDEX_FILENAME = "chunk_index_file"
SENTENCE_INDEX_FILENAME = "sentence_index_file"
PATHS_FILENAME = "paths_file"
MODIFIED_TIMES_FILE = "modified_times_file"

def load_questions_and_answers(file_path):
    q_and_a = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            q_and_a.append((row['question_details'], row['answers']))
    return q_and_a

def load_modified_times():
    if os.path.exists(MODIFIED_TIMES_FILE):
        with open(MODIFIED_TIMES_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return {}

def save_modified_times(modified_times):
    with open(MODIFIED_TIMES_FILE, "wb") as f:
        pickle.dump(modified_times, f)

def save_indices_and_paths(text_index, sentence_index, text_paths, sentence_paths, image_paths, chunk_index_file, sentence_index_file, paths_file):
    faiss.write_index(text_index, chunk_index_file)
    faiss.write_index(sentence_index, sentence_index_file)
    with open(paths_file, "wb") as f:
        pickle.dump((text_paths, sentence_paths, image_paths), f)

def load_indices_and_paths(chunk_index_file, sentence_index_file, paths_file):
    text_index = faiss.read_index(chunk_index_file)
    sentence_index = faiss.read_index(sentence_index_file)
    with open(paths_file, "rb") as f:
        text_paths, sentence_paths, image_paths = pickle.load(f)
    return text_index, sentence_index, text_paths, sentence_paths, image_paths

def preprocess_documents(root_folder: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, PIL.Image.Image]]]:
    text_documents = []
    image_documents = []
    sentence_documents = []

    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            input_type = file.split('.')[-1]

            if input_type not in ['pdf', 'pptx', 'docx','jpg', 'png', 'jpeg']:
                continue

            if input_type in ['pdf', 'pptx', 'docx']:
                text = analyze_input(input_type, None, file_path)
                cleaned_text = processing.clean_text(text)
                chunks = processing.split_text(cleaned_text)
                for chunk in chunks:
                    text_documents.append((file_path, chunk))
            elif input_type in ['jpg', 'png', 'jpeg']:
                image = PIL.Image.open(file_path)
                image_documents.append((file_path, image))

    for file_path, chunk in text_documents:
        sentences = processing.split_into_sentences(chunk)
        for sentence in sentences:
            sentence_documents.append((file_path, sentence))

    return text_documents, sentence_documents, image_documents

def search(query: str, index: faiss.IndexIDMap, text_paths: Dict[int, str], image_paths: Dict[int, str]) -> List[Tuple[str, float]]:
    # Embed the query
    query_embedding = embeddings.embed_text(query)

    # Search the index
    D, I = index.search(np.array([query_embedding]), k=3)

    # Get the paths and scores of the search results
    results = []
    for i, score in zip(I[0], D[0]):
        if i in text_paths:
            path, chunk = text_paths[i]
            results.append((path, chunk, score))
        elif i in image_paths:
            results.append((image_paths[i], "", score))  # Add an empty string for the chunk in the case of images

    return results

def analyze_input(input_type, company, url):
    text = ""
    if input_type == "url":
        data = urlscrape.link(url)
    elif input_type in ["pdf", "pptx", "docx"]:
        file_path = url

        with open(file_path, "rb") as file:
            if input_type == "pdf":
                text = readpdf.read_pdf(file)
            elif input_type == "docx":
                text = readdoc.read_word(file_path)
            elif input_type == "pptx":
                file_content = file.read()
                text = readppt.read_ppt(file_content)
        data = text
    else:
        raise ValueError("Invalid input type")

    return data

def generate_questions(query: str, model_name: str, tokenizer) -> List[str]:
    prompt = f"Given the context '{query}', What 1-2 questions should we ask to gain more information?"
    messages = [{"role": role, "content": content} for role, content in [("system", "You are a helpful assistant."), ("user", prompt)]]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.75
    )

    questions = response.choices[0]['message']['content'].strip().split("\n")
    questions = [question.strip() for question in questions if question.strip()]

    return questions

def main():
    current_dir = os.getcwd()
    root_folder = os.path.join(current_dir, "Input")
    # root_folder = input("Please input the filepath to the folder:- ")
    q_and_a = load_questions_and_answers('Input/questions.csv')

    # Load the old modified times
    old_modified_times = load_modified_times()

    # Get the new modified times
    new_modified_times = {}
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            new_modified_times[file_path] = os.path.getmtime(file_path)

    # Check if any files have been added, modified, or deleted
    files_have_changed = len(old_modified_times) != len(new_modified_times)
    if not files_have_changed:
        for file_path, new_time in new_modified_times.items():
            if file_path not in old_modified_times or old_modified_times[file_path] != new_time:
                files_have_changed = True
                break

    # Check if any of the previously indexed files have been deleted
    _, _, text_paths, _, _ = load_indices_and_paths(CHUNK_INDEX_FILENAME, SENTENCE_INDEX_FILENAME, PATHS_FILENAME)
    for file_path in text_paths:
        if not os.path.isfile(file_path):
            files_have_changed = True
            break

    # If any files have changed, re-embed and re-index them
    if files_have_changed:
        print("Files have changed. Re-embedding and re-indexing...")
        text_documents, sentence_documents, image_documents = preprocess_documents(root_folder)
        text_index, sentence_index, text_paths, sentence_paths, image_paths = embeddings.index_embeddings(text_documents, image_documents)
        save_indices_and_paths(text_index, sentence_index, text_paths, sentence_paths, image_paths, CHUNK_INDEX_FILENAME, SENTENCE_INDEX_FILENAME, PATHS_FILENAME)
    else:
        print("No files have changed. Loading the existing index...")
        try:
            text_index, sentence_index, text_paths, sentence_paths, image_paths = load_indices_and_paths(CHUNK_INDEX_FILENAME, SENTENCE_INDEX_FILENAME, PATHS_FILENAME)
        except Exception as e:
            print(f"Could not load index and paths due to error: {str(e)}. Creating a new index...")
            text_index, sentence_index, text_paths, sentence_paths, image_paths = embeddings.index_embeddings(text_documents, image_documents)
            save_indices_and_paths(text_index, sentence_index, text_paths, sentence_paths, image_paths, CHUNK_INDEX_FILENAME, SENTENCE_INDEX_FILENAME, PATHS_FILENAME)

    # Save the new modified times
    save_modified_times(new_modified_times)
    query = input("Please input your query:- ")
    section_names = input("Please input the names of the report sections (comma-separated):- ").split(',')

    report = {}
    for section in section_names:
        print(f"\nGenerating content for {section}...\n")
        questions = generate_questions(query, MODEL, tokenizer)
        print(f"The questions are: {questions}")
        section_content = []

        for question in questions:
            # Search for answers to the question in the Q&A pairs
            relevant_facts = [answer for q, answer in q_and_a if q == question]
            sentence_results = search(question, sentence_index, sentence_paths, image_paths)
            results = search(question, text_index, text_paths, image_paths)
            print(f"\n The results are: {results}\n\n")
            # Consider both the results from the documents and the relevant facts from the Q&A pairs
            all_results = [chunk for path, chunk, score in results] + relevant_facts
            for result in all_results:
                text = gpt_calls.recursive_analyze(result)
                summary = gpt_calls.recursive_analyze(text)
                section_content.append(summary)
            # for i in range(min(1, len(results))):
            #     top_result_path, top_result_chunk = results[i][:2]
            #     input_type = top_result_path.split('.')[-1]
            #     # Should combine path analysis + the chunk analysis from the following line
            #     # Should add a lookup to the actual path so we get the "quotes" which can be inserted into the document if needed
            #     # text = analyze_input(input_type, None, top_result_path)
            #     text = gpt_calls.recursive_analyze(top_result_chunk)
            #     summary = gpt_calls.recursive_analyze(text)
            #     section_content.append(summary)

        report[section] = ' '.join(section_content)

    for section, content in report.items():
        print(f"\n{section}\n{content}")

if __name__ == "__main__":
    main()
    