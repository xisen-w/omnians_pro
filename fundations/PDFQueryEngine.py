from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from llama_index.core import VectorStoreIndex

class PDFQueryEngine:
    def __init__(self, llmsherpa_api_url):
        """
        Initialize the PDFQueryEngine with the LLM Sherpa API URL.

        Args:
            llmsherpa_api_url (str): The API URL for LLM Sherpa.
        """
        self.llmsherpa_api_url = llmsherpa_api_url
        self.documents = None
        self.index = None
        self.query_engine = None

    def load_pdf(self, pdf_source):
        """
        Load the PDF document using SmartPDFLoader.

        Args:
            pdf_source (str): The URL or file path to the PDF.

        Returns:
            list: A list of documents loaded from the PDF.
        """
        try:
            pdf_loader = SmartPDFLoader(llmsherpa_api_url=self.llmsherpa_api_url)
            self.documents = pdf_loader.load_data(pdf_source)
            print("PDF successfully loaded.")
            return self.documents
        except Exception as e:
            print(f"An error occurred while loading the PDF: {e}")
            return None

    def create_index(self):
        """
        Create an index from the loaded documents.
        """
        if self.documents:
            try:
                self.index = VectorStoreIndex.from_documents(self.documents)
                self.query_engine = self.index.as_query_engine()
                print("Index successfully created.")
            except Exception as e:
                print(f"An error occurred while creating the index: {e}")
        else:
            print("No documents loaded. Please load a PDF first.")

    def query(self, question):
        """
        Query the index using the provided question.

        Args:
            question (str): The question to query the PDF contents.

        Returns:
            str: The response from the query engine.
        """
        if self.query_engine:
            try:
                response = self.query_engine.query(question)
                return response
            except Exception as e:
                print(f"An error occurred while querying the index: {e}")
                return None
        else:
            print("Query engine not initialized. Please create an index first.")
            return None


if __name__ == "__main__":
    # Example usage
    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_url = "https://arxiv.org/pdf/1910.13461.pdf"  # URL to a PDF file

    print("Initializing PDFQueryEngine...")

    # Initialize the PDFQueryEngine
    pdf_query_engine = PDFQueryEngine(llmsherpa_api_url)

    print("Query engine Done. Loading PDFs")

    # Load the PDF document
    documents = pdf_query_engine.load_pdf(pdf_url)

    print("Creating index")

    # Create an index from the loaded documents
    pdf_query_engine.create_index()

    print("Querying!")

    # Query the index
    response = pdf_query_engine.query("list all the tasks that work with bart")
    print(response)

    response = pdf_query_engine.query("what is the bart performance score on squad")
    print(response)




