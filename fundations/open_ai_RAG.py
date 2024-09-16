import os
import openai
import pandas as pd
from scipy import spatial
import numpy as np
import sys
import PyPDF2  # To extract text from PDF
from typing import Union, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fundations.foundation import LLMResponse

# Get the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = openai.OpenAI(api_key=api_key)

# Updated embedding models
EMBEDDING_MODEL = "text-embedding-3-small"  # Updated model
GPT_MODEL = "gpt-4o-mini"

# Function to normalize embeddings
def normalize_l2(x):
    x = np.array(x)
    norm = np.linalg.norm(x)
    return x if norm == 0 else x / norm

class Retriever:
    def __init__(self):
        # DataFrame to store text and embeddings
        self.df = pd.DataFrame(columns=["text", "embedding"])

    def embed_text(self, text: str) -> list:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def add_to_index(self, text: str):
        """
        Adds text and its embedding to the index (DataFrame).
        """
        embedding = self.embed_text(text)
        new_entry = pd.DataFrame([[text, embedding]], columns=["text", "embedding"])
        self.df = pd.concat([self.df, new_entry], ignore_index=True)

    def vector_search(self, query: str, top_n: int = 1) -> list:
        """
        Perform a vector search by embedding the query and finding the most relevant text.
        Returns the top_n most related texts and their similarity scores.
        """
        # Embed the query
        query_embedding = self.embed_text(query)

        # Calculate similarity between query and stored embeddings
        self.df["similarity"] = self.df["embedding"].apply(
            lambda x: 1 - spatial.distance.cosine(query_embedding, x)
        )

        # Sort by similarity and return the top_n texts
        results = self.df.sort_values(by="similarity", ascending=False).head(top_n)
        return results[["text", "similarity"]].values.tolist()

    def ask_gpt(self, query: str, context: str):
        """
        Use GPT to answer the query, with relevant context inserted.
        """
        llm = LLMResponse(model_name=GPT_MODEL)
        system_prompt = "You answer questions using the provided context."
        user_prompt = f"Context: {context}\n\nQuestion: {query}"
        
        response = llm.llm_output(user_prompt, system_prompt)
        return response.content

    def retrieve_and_ask(self, query: str, top_n: int = 5):
        """
        Perform a vector search for relevant text and then ask GPT the question with context.
        """
        # Perform vector search
        top_texts = self.vector_search(query, top_n=top_n)
        context = "\n\n".join([text for text, _ in top_texts])

        # Ask GPT with the retrieved context
        answer = self.ask_gpt(query, context)
        return answer, context

    def chunking(self, text: str, mode: str = "sentence"):
        """
        Splits the text into chunks based on the chosen mode ('sentence', 'paragraph', or 'page').
        """
        if mode == "sentence":
            return text.split(". ")
        elif mode == "paragraph":
            return text.split("\n\n")
        elif mode == "page":
            return text.split("\f")
        else:
            raise ValueError("Invalid mode! Choose either 'sentence', 'paragraph', or 'page'.")

    def create_embedding_for_pdf(self, pdf_path: str, chunk_mode: str = "paragraph"):
        """
        Extracts text from a PDF, chunks it, and generates embeddings for each chunk.
        """
        # Step 1: Extract text from PDF
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()

        # Step 2: Chunk the text
        chunks = self.chunking(pdf_text, mode=chunk_mode)

        # Step 3: Create embeddings for each chunk and index them
        for chunk in chunks:
            # Split the chunk further if it's too large
            sub_chunks = self.split_text(chunk, max_tokens=8000)  # Leave some buffer
            for sub_chunk in sub_chunks:
                self.add_to_index(sub_chunk)

    def rag_complete(self, pdf_path: str, query: str, chunk_mode: str = "paragraph"):
        """
        Given a PDF and a query, generate a response using retrieval-augmented generation (RAG).
        """
        # Step 1: Create embeddings for the PDF
        self.create_embedding_for_pdf(pdf_path, chunk_mode)

        # Step 2: Retrieve the most relevant chunks and pass them to GPT
        answer, context = self.retrieve_and_ask(query)
        
        # Debug print
        print("Debug - Retrieved context:")
        print(context)
        print("\nDebug - Generated answer:")
        print(answer)

        return answer, context

    def split_text(self, text, max_tokens=8000):
        tokens = self.tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for token in tokens:
            if current_length + len(token) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [token]
                current_length = len(token)
            else:
                current_chunk.append(token)
                current_length += len(token)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def tokenize(self, text):
        # Implement a simple tokenization method
        # This is a basic example; you might want to use a more sophisticated tokenizer
        return text.split()
    

class Citation_Retriever(Retriever):
    def __init__(self):
        super().__init__()
        self.df = pd.DataFrame(columns=["text", "embedding", "source", "page"])

    def add_to_index(self, text: str, source: str, page: int):
        embedding = self.embed_text(text)
        new_entry = pd.DataFrame([[text, embedding, source, page]], columns=["text", "embedding", "source", "page"])
        self.df = pd.concat([self.df, new_entry], ignore_index=True)

    def vector_search(self, query: str, top_n: int = 1) -> list:
        query_embedding = self.embed_text(query)
        self.df["similarity"] = self.df["embedding"].apply(
            lambda x: 1 - spatial.distance.cosine(query_embedding, x)
        )
        results = self.df.sort_values(by="similarity", ascending=False).head(top_n)
        return results[["text", "similarity", "source", "page"]].values.tolist()

    def retrieve_and_ask(self, query: str, top_n: int = 5):
        top_texts = self.vector_search(query, top_n=top_n)
        context = "\n\n".join([f"[Source: {source}, Page: {page}]\n{text}" for text, _, source, page in top_texts])
        answer = self.ask_gpt(query, context)
        return answer, context, top_texts

    def create_embedding_for_pdf(self, pdf_path: str, chunk_mode: str = "paragraph"):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                chunks = self.chunking(page_text, mode=chunk_mode)
                for chunk in chunks:
                    sub_chunks = self.split_text(chunk, max_tokens=8000)
                    for sub_chunk in sub_chunks:
                        self.add_to_index(sub_chunk, os.path.basename(pdf_path), page_num)

    def rag_complete(self, pdf_paths: Union[str, List[str]], query: str, chunk_mode: str = "paragraph"):
        """
        Given one or more PDFs and a query, generate a response using retrieval-augmented generation (RAG).
        
        :param pdf_paths: A single PDF path (str) or a list of PDF paths (List[str])
        :param query: The query to answer
        :param chunk_mode: The chunking mode for text extraction
        :return: A tuple containing the answer, context, and top relevant texts
        """
        # Convert single PDF path to a list if necessary
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        
        # Process each PDF
        for pdf_path in pdf_paths:
            self.create_embedding_for_pdf(pdf_path, chunk_mode)
        
        # Retrieve relevant information and generate answer
        answer, context, top_texts = self.retrieve_and_ask(query)
        
        # Debug prints
        print("Debug - Retrieved context:")
        print(context)
        print("\nDebug - Generated answer:")
        print(answer)
        print("\nDebug - Top relevant texts:")
        for text, similarity, source, page in top_texts:
            print(f"Source: {source}, Page: {page}, Similarity: {similarity:.4f}")
            print(f"Text: {text[:100]}...")  # Print first 100 characters of each text

        return answer, context, top_texts

# Example usage
if __name__ == "__main__":
    retriever = Citation_Retriever()
    
    pdf_path = "/Users/wangxiang/Desktop/omnians_pro/test/demo_reading/title-all-about-iraq-re-modifying-older-slogans-and-chants-in-tishreen-october-protests-author-author-mustafa.pdf"
    query = "What are the sources of chant?"
    
    rag_result, rag_context, top_texts = retriever.rag_complete(pdf_path, query)
    print("\nFinal RAG Answer:", rag_result)
    print("\nTop relevant texts:")
    for text, similarity, source, page in top_texts:
        print(f"Source: {source}, Page: {page}, Similarity: {similarity:.4f}")
        print(f"Text: {text[:100]}...")  # Print first 100 characters of each text

# # Example usage
# if __name__ == "__main__":
#     retriever = Retriever()

#     # Add texts to the index
#     retriever.add_to_index("Curling at the 2022 Winter Olympics was held in Beijing.")
#     retriever.add_to_index("In the men's curling event, the gold medal was won by Sweden.")
#     retriever.add_to_index("In the women's curling event, the gold medal was won by Great Britain.")
#     retriever.add_to_index("In the mixed doubles curling event, the gold medal was won by Italy.")

#     # Perform a search and ask GPT
#     result = retriever.retrieve_and_ask("Who won the gold medal in men's curling at the 2022 Winter Olympics?")
#     print("Answer:", result)

#     # Test chunking and creating embeddings for a PDF
#     retriever.create_embedding_for_pdf("/Users/wangxiang/Desktop/omnians_pro/test/demo_reading/title-all-about-iraq-re-modifying-older-slogans-and-chants-in-tishreen-october-protests-author-author-mustafa.pdf", chunk_mode="paragraph")
    
#     # Test RAG complete functionality
#     rag_result, rag_context = retriever.rag_complete("/Users/wangxiang/Desktop/omnians_pro/test/demo_reading/title-all-about-iraq-re-modifying-older-slogans-and-chants-in-tishreen-october-protests-author-author-mustafa.pdf", "What are the sources of chant?")
#     print("\nFinal RAG Answer:", rag_result)
#     print("\nRetrieved Context:", rag_context)
