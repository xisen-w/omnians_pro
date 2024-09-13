from openai import OpenAI # Assuming OpenAI's library is being used, install with `pip install openai`
from pydantic import BaseModel
from fundations.foundation import LLMResponse
import os

os.environ['OPENAI_API_KEY'] = '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'

class LLMResponsePro(LLMResponse):
    def __init__(self, model_name):
        """
        Initialize the LLMResponse with the given model name.
        """
        self.model_name = model_name #Eg "gpt-4o-2024-08-06"
        self.client = OpenAI()
        # Ensure your API key is set in the environment

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

