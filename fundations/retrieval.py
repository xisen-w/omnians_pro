import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load a model and tokenizer to generate embeddings (e.g., a BERT-like model)
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text: str):
    """Generate embeddings for the input text using the pre-trained model."""
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    except Exception as e:
        logger.error(f"Error generating embeddings for text: {text} - {e}")
        return None

# Sample document corpus
corpus = [
    "AI agents are transforming the field of autonomous systems.",
    "Machine learning models for autonomous research have limitations in scalability.",
    "New AI-driven models in autonomous systems show improvements over traditional models.",
    "Criticism of AI agent architectures focuses on their lack of generalization."
]

# Create embeddings for all documents
corpus_embeddings = []
try:
    corpus_embeddings = np.array([get_embedding(doc)[0] for doc in corpus if get_embedding(doc) is not None])
    if len(corpus_embeddings) == 0:
        raise ValueError("No embeddings were generated for the corpus. Check your input or embedding model.")
except Exception as e:
    logger.error(f"Error while creating document embeddings: {e}")
    exit(1)

# Build a FAISS index for efficient vector search
try:
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    index.add(corpus_embeddings)  # Add document embeddings to the index
except Exception as e:
    logger.error(f"Error building FAISS index: {e}")
    exit(1)

def retrieve_documents(query: str, k: int = 3):
    """Retrieve top-k most similar documents to the query using FAISS."""
    try:
        query_embedding = get_embedding(query)
        if query_embedding is None:
            raise ValueError(f"Failed to generate embeddings for the query: {query}")
        
        distances, indices = index.search(np.array([query_embedding[0]]), k)  # Perform the search
        retrieved_docs = [corpus[i] for i in indices[0]]
        return retrieved_docs, distances[0]
    except Exception as e:
        logger.error(f"Error during document retrieval: {e}")
        return [], []

# Example query
if __name__ == "__main__":
    try:
        query = "What are the limitations of machine learning models in research?"
        retrieved_docs, distances = retrieve_documents(query)
        
        if retrieved_docs:
            logger.info(f"Retrieved Documents: {retrieved_docs}")
        else:
            logger.info(f"No documents were retrieved for the query: {query}")
    except Exception as e:
        logger.error(f"Error in the main query process: {e}")