import sys
import os
import random

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fundations.dataUploader import DataUploader  # Corrected spelling
from Agents.basicAgents import LLMAgent  # Importing LLMAgent from your existing infrastructure
import json

class PDFSummaryAgent(LLMAgent):
    def __init__(self, model_name="gpt-4o-mini"):
        super().__init__(model_name)  # Remove use_pro from here
        self.data_uploader = DataUploader()
        self.max_words = 3000  # Define the maximum word count limit (can adjust this based on GPT token limit)

    def random_adjust(self, text_content):
        """
        Adjust the content by randomly removing sentences until it is within the word limit.
        Args:
            text_content (str): The content of the PDF as a string.

        Returns:
            str: The adjusted content with word count within the limit.
        """
        sentences = text_content.split(". ")
        total_words = len(" ".join(sentences).split())
        
        # Log the initial state
        print(f"Initial number of sentences: {len(sentences)}")
        print(f"Initial word count: {total_words}")
        
        while total_words > self.max_words and len(sentences) > 1:
            random_index = random.randint(0, len(sentences) - 1)  # Choose a random sentence to delete
            print(f"Deleting sentence {random_index}: {sentences[random_index]}")  # Log which sentence is being deleted
            del sentences[random_index]  # Delete that sentence
            
            total_words = len(" ".join(sentences).split())  # Recalculate word count
            print(f"New word count after deletion: {total_words}")
        
        if total_words == 0:
            print("Warning: Word count reached zero after adjustment.")
        
        return ". ".join(sentences)

    def summarize_pdf(self, pdf_path=None, pdf_url=None):
        """
        Summarize the content of a PDF file.

        Args:
            pdf_path (str): Path to the local PDF file.
            pdf_url (str): URL to the PDF file.

        Returns:
            str: The summary of the PDF content.
        """
        # Load the PDF from a URL or file path
        if pdf_url:
            content = self.data_uploader.upload_from_url(pdf_url)
            if content:
                text_content = self.data_uploader.parse_html()  # Use parse_html to extract text from HTML
            else:
                text_content = None
        elif pdf_path:
            text_content = self.data_uploader.upload_from_pdf(pdf_path)
        else:
            raise ValueError("Either a PDF path or URL must be provided.")

        # Ensure content was successfully extracted
        if not text_content:
            return "Failed to extract content from the PDF."

        # Check the word count
        word_count = len(text_content.split())
        print(f"Word count: {word_count}")
        if word_count > self.max_words:
            print(f"Content exceeds the maximum word limit of {self.max_words}. Adjusting content...")
            text_content = self.random_adjust(text_content)
            print(f"Adjusted word count: {len(text_content.split())}")

        try:
            # Generate a summary using the LLM
            system_prompt = """
            You are a professional summarizer. Your task is to provide a concise and clear summary of the following content:

            The summary should cover the main points, key arguments, and conclusions. Keep it brief but informative.
        
            Format: Title. Main Narrative. Insight.
            """

            user_prompt = text_content

            # Use the perform_action method from LLMAgent
            response = self.perform_action(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema_class=None
            )

            return response.content if response else "No summary generated."
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

    def to_string(self):
        """
        Convert the LLM response to a readable string.
        """
        if self.previous_result:
            return json.dumps(self.previous_result.dict(), indent=2)
        return "No result available."

# Example Usage
if __name__ == "__main__":
    agent = PDFSummaryAgent(model_name="gpt-4o-mini")

    # Summarize PDF from URL
    # pdf_url = "https://.org/pdf/1910.13461.pdf"
    # summary = agent.summarize_pdf(pdf_url=pdf_url)
    # print("Summary from URL:")
    # print(summary)

    # Summarize PDF from local file
    pdf_path = "/Users/wangxiang/Desktop/omnians_pro/test/readings/science1.pdf"
    summary = agent.summarize_pdf(pdf_path=pdf_path)
    print("Summary from Local File:")
    print(summary)