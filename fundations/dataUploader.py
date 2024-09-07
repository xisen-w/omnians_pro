import requests
from bs4 import BeautifulSoup
import PyPDF2
import fitz  # PyMuPDF

class DataUploader:
    def __init__(self):
        """
        Initialize the DataUploader class.
        """
        self.html_content = None
        self.pdf_text = None

    def upload_from_url(self, url):
        """
        Fetches HTML content from a URL and stores it in the instance.

        Args:
            url (str): The URL of the webpage to fetch.

        Returns:
            str: The HTML content of the page.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
            self.html_content = response.text
            print("HTML content successfully fetched.")
            return self.html_content
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching the HTML content: {e}")
            return None

    def parse_html(self):
        """
        Parses the HTML content using BeautifulSoup and returns the text.

        Returns:
            str: The parsed text from the HTML content.
        """
        if self.html_content:
            soup = BeautifulSoup(self.html_content, 'html.parser')
            text = soup.get_text()
            return text
        else:
            print("No HTML content available to parse.")
            return None

    # def upload_from_pdf(self, pdf_file_path):
    #     """
    #     Extracts text from a PDF file and stores it in the instance.

    #     Args:
    #         pdf_file_path (str): The file path to the PDF file.

    #     Returns:
    #         str: The extracted text from the PDF.
    #     """
    #     try:
    #         with open(pdf_file_path, 'rb') as file:
    #             reader = PyPDF2.PdfReader(file)
    #             self.pdf_text = ""
    #             for page in reader.pages:
    #                 self.pdf_text += page.extract_text()
    #         print("PDF text successfully extracted.")
    #         return self.pdf_text
    #     except Exception as e:
    #         print(f"An error occurred while reading the PDF file: {e}")
    #         return None
        
    

    def upload_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF using PyMuPDF.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            str: Extracted text content.
        """
        try:
            doc = fitz.open(pdf_path)  # Open the PDF
            text_content = ""
            for page in doc:
                text_content += page.get_text()  # Extract text from each page
            doc.close()
            return text_content
        except Exception as e:
            print(f"An error occurred while extracting the PDF: {e}")
            return None

    def get_html_content(self):
        """
        Returns the stored HTML content.

        Returns:
            str: The stored HTML content.
        """
        return self.html_content

    def get_pdf_text(self):
        """
        Returns the stored PDF text.

        Returns:
            str: The stored text extracted from the PDF.
        """
        return self.pdf_text


# Example usage
if __name__ == "__main__":
    uploader = DataUploader()

    # Example of fetching HTML content from a URL
    html_content = uploader.upload_from_url("https://arxiv.org/html/2306.14565v4")
    parsed_text = uploader.parse_html()
    print(parsed_text)

    # # Example of uploading and extracting text from a PDF file
    pdf_text = uploader.upload_from_pdf("/Users/wangxiang/Desktop/omnians_pro/test/readings/science1.pdf")
    print(pdf_text)