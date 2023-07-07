import re
import nltk
from typing import List, Tuple
from transformers import GPT2Tokenizer
from nltk import tokenize

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
CHUNK_SIZE=1000

def clean_text(text):
    cleaned_text = " ".join(text.split())
    cleaned_text = re.sub(r'http\S+', '', cleaned_text)
    cleaned_text = re.sub(r'<script.*?>.*?</script>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<style.*?>.*?</style>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = " ".join(cleaned_text.split())
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,!?/:;()%$@&\s]', '', cleaned_text)
    cleaned_text = re.sub(r'(?i)(terms\s*and\s*conditions|privacy\s*policy|copyright|blog|legal|careers|cdn*).{0,10}', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def split_text(text: str, max_tokens=CHUNK_SIZE) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = tokenizer(sentence)["input_ids"]
        # Exclude the special tokens ([CLS], [SEP]) from the token count
        sentence_token_count = len(sentence_tokens) - 2

        if current_tokens + sentence_token_count > max_tokens:
            # Exceeds token limit, create a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_token_count
        else:
            # Append the sentence to the current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_token_count

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def split_into_sentences(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    return sentences