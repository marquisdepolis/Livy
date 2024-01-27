import torch
import torchvision.models as models
import torchvision.transforms as transforms
import faiss
import openai
import numpy as np
import processing
from typing import List, Tuple, Dict
from PIL import Image

# Initialize the pre-trained ResNet model
resnet = models.resnet18(pretrained=True)
resnet.eval()

# Define the preprocessing transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# For image embedding
def embed_image(image: Image.Image) -> np.ndarray:
    # Apply the transformations to the image
    transformed_image = transform(image).unsqueeze(0)

    # Extract the features using the ResNet model
    with torch.no_grad():
        features = resnet(transformed_image).numpy()

    return features

# Embed the text
def embed_text(text: str) -> np.ndarray:
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding'])

def index_embeddings(text_documents: List[Tuple[str, str]], image_documents: List[Tuple[str, Image.Image]]) -> Tuple[faiss.Index, Dict[int, Tuple[str, str]], Dict[int, str]]:
    text_paths = {}
    image_paths = {}
    sentence_paths = {}

    # Obtain the first text embedding to get its dimensions
    first_path, first_text = text_documents[0]
    first_embedding = embed_text(first_text)
    embedding_dim = first_embedding.shape[0]

    # Initialize the index with the dynamic embedding dimension
    text_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))  # Index for chunks
    sentence_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))  # Index for sentences

    # Add the first text embedding to the index
    text_index.add_with_ids(first_embedding.reshape(1, -1), np.array([0]))
    text_paths[0] = (first_path, first_text)  # Store the path and chunk together

    current_id_text = 1
    current_id_sentence = 0

    # Add the remaining text embeddings to the index
    for path, text in text_documents[1:]:
        chunks = processing.split_text(text)
        for chunk in chunks:
            print(f"Embedding chunk with {len(chunk)} tokens")
            embedding = embed_text(chunk)
            text_index.add_with_ids(embedding.reshape(1, -1), np.array([current_id_text]))
            text_paths[current_id_text] = (path, chunk)  # Store the path and chunk together
            current_id_text += 1

        sentences = processing.split_into_sentences(text)
        for sentence in sentences:
            print("Embedding sentence")
            sentence_embedding = embed_text(sentence)
            sentence_index.add_with_ids(sentence_embedding.reshape(1, -1), np.array([current_id_sentence]))
            sentence_paths[current_id_sentence] = (path, sentence)
            current_id_sentence += 1

    # Add the image embeddings to the index
    for path, image in image_documents:
        print("Embedding image")
        embedding = embed_image(image)
        text_index.add_with_ids(embedding.reshape(1, -1), np.array([current_id_text]))  # Here I'm assuming you want the image embeddings to be in the chunk index.
        image_paths[current_id_text] = path
        current_id_text += 1

    return text_index, sentence_index, text_paths, sentence_paths, image_paths