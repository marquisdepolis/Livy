# %%
import spacy
from functools import lru_cache
import os
import re
import warnings
from urllib.parse import urlparse, urljoin
import requests
import tldextract
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
from tenacity import retry, stop_after_attempt, wait_random_exponential

warnings.filterwarnings("ignore")
ALLOWED_TLDS = {"com","app", "org", "net", "edu", "gov"}
EXCLUDED_KEYWORDS = {
    "login",
    "signup",
    "results",
    "search",
    "register",
    "account",
    "privacy",
    "terms",
    "policy",
    "disclaimer",
    "jobs",
    "careers",
    "blog"
    "contact",
    "cookie",
    "support",
    "forum",
    "cdn",
    "newsletter",
    "status",
}

def get_links(soup, base_url):
    links = set()
    parsed_base_url = urlparse(base_url)
    ext = tldextract.extract(parsed_base_url.netloc)
    base_domain = f"{ext.domain}.{ext.suffix}"
    allowed_domains = {base_domain}

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if not href.startswith("http"):
            href = urljoin(base_url, href)
        
        parsed_url = urlparse(href)
        ext = tldextract.extract(parsed_url.netloc)
        domain = f"{ext.domain}.{ext.suffix}"

        if (base_url in href
            and domain in allowed_domains
            and ext.suffix in ALLOWED_TLDS
            and not any(keyword in href.lower() for keyword in EXCLUDED_KEYWORDS)
        ):
            links.add(href)
    print(links)
    return links

def fetch_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {url}\n{str(e)}")
        return None
    return soup

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

def link(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.hostname:
        print("Invalid URL. Please provide a valid URL with a scheme (e.g., http:// or https://).")
        return None
    base_url = parsed_url.scheme + "://" + parsed_url.hostname
    soup = fetch_html(url)
    if not soup:
        return None
    links = get_links(soup, base_url)
    all_text = []
    for link in links:
        sub_soup = fetch_html(link)
        if sub_soup:
            text = clean_text(sub_soup.get_text())
            all_text.append(text)
    full_text = " ".join(all_text)
    full_text = clean_text(full_text)
    return full_text