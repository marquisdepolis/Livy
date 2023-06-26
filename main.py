import pickle
import nltk
import PIL
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import concurrent.futures
import json
import os
import re
import warnings
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple, Dict
from urllib.parse import urlparse, urljoin

import faiss
import numpy as np
import openai
import PyPDF2
import requests
import spacy
import tldextract
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pptx import Presentation
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tkinter import Tk, filedialog
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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

MODEL = "gpt-4"
CHUNK_SIZE=7000
INDEX_FILENAME = "index_file"
PATHS_FILENAME = "paths_file"

def save_index_and_paths(index, text_paths, image_paths, index_file, paths_file):
    faiss.write_index(index, index_file)
    with open(paths_file, "wb") as f:
        pickle.dump((text_paths, image_paths), f)

def load_index_and_paths(index_file, paths_file):
    index = faiss.read_index(index_file)
    with open(paths_file, "rb") as f:
        text_paths, image_paths = pickle.load(f)
    return index, text_paths, image_paths

def preprocess_documents(root_folder: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, PIL.Image.Image]]]:
    text_documents = []
    image_documents = []

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

    return text_documents, image_documents

def search(query: str, index: faiss.IndexIDMap, text_paths: Dict[int, str], image_paths: Dict[int, str]) -> List[Tuple[str, float]]:
    # Embed the query
    query_embedding = embeddings.embed_text(query)

    # Search the index
    D, I = index.search(np.array([query_embedding]), k=10)

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
    openai.api_key = os.getenv("OPENAI_API_KEY")

    root_folder = input("Please input the filepath to the folder:- ")
    text_documents, image_documents = preprocess_documents(root_folder)

    try:
        index, text_paths, image_paths = load_index_and_paths(INDEX_FILENAME, PATHS_FILENAME)
    except Exception as e:
        print(f"Could not load index and paths due to error: {str(e)}. Creating a new index...")
        index, text_paths, image_paths = embeddings.index_embeddings(text_documents, image_documents)
        save_index_and_paths(index, text_paths, image_paths, INDEX_FILENAME, PATHS_FILENAME)

    query = input("Please input your query:- ")
    section_names = input("Please input the names of the report sections (comma-separated):- ").split(',')

    report = {}
    for section in section_names:
        print(f"\nGenerating content for {section}...\n")
        questions = generate_questions(query, MODEL, tokenizer)
        print(f"The questions are: {questions}")
        section_content = []

        for question in questions:
            results = search(question, index, text_paths, image_paths)
            print(f"\n The results are: {results}\n\n")
            for i in range(min(1, len(results))):
                top_result_path, top_result_chunk = results[i][:2]
                input_type = top_result_path.split('.')[-1]
                text = analyze_input(input_type, None, top_result_path)
                summary = gpt_calls.recursive_analyze(text)
                section_content.append(summary)

        report[section] = ' '.join(section_content)

    for section, content in report.items():
        print(f"\n{section}\n{content}")

if __name__ == "__main__":
    main()
