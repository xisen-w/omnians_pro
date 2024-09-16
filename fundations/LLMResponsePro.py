from openai import OpenAI
from pydantic import BaseModel
from fundations.foundation import LLMResponse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMResponsePro(LLMResponse):
    def __init__(self, model_name):
        """
        Initialize the LLMResponse with the given model name.
        """
        self.model_name = model_name  # Eg "gpt-4o-2024-08-06"
        
        # Get the API key from environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)

    def structured_output(self, schema_class, user_prompt, system_prompt):
        """
        Structure the output according to the provided schema, user prompt, and system prompt.
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=schema_class,
            )

            response = completion.choices[0].message.parsed
            return response

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

