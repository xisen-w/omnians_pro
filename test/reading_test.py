from llama_index.readers.smart_pdf_loader import SmartPDFLoader

# Define the API URL and the path to your PDF
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_path = "/Users/wangxiang/Desktop/omnians_pro/test/readings/science1.pdf"  # Replace with your actual PDF file path

# Initialize the PDF loader
pdf_loader = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url)

# Load and parse the PDF document
documents = pdf_loader.load_data(pdf_path)

